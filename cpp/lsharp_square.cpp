// Passive L#-square in C++: the algorithm and CNF of ObservationTreeSquare.py and DfaEncoding.py,
// with CaDiCaL linked in-process. See those files for an explanation of the algorithm and encoding.
//
// Usage: lsharp_square FILE [--dfa]
//
// FILE is a sample in Abbadingo format; lines with labels other than 0 and 1 are skipped. Prints
// "file,states,agrees,solver_time,total_time" and, with --dfa, then one line per state:
// "accepting target_0 target_1 ..." (state 0 is initial).
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cadical.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// Same constants as in the Python implementation.
constexpr int kPairwiseAtMostOneLimit = 32;
constexpr int kExtraStatesPerEncoding = 4;

struct Sample {
    std::vector<int> word;
    bool accepts;
};

struct Hypothesis {
    std::vector<std::vector<int>> transitions;  // Target per state and letter.
    std::vector<bool> accepting;
};

// Observation tree restricted to its informative nodes, numbered breadth-first.
struct ObservationTree {
    int alphabet_size = 0;
    std::vector<int> parent, letter;  // parent[0] = -1.
    std::vector<int> output;          // 1 accepting, -1 rejecting, 0 unknown.
    // Per suffix (0 = empty): the pairs (u, output of uw); per node u: the pairs (w, output of uw).
    std::vector<std::vector<std::pair<int, int>>> suffix_occurrences, below;

    ObservationTree(int alphabet, const std::vector<Sample>& samples) : alphabet_size(alphabet) {
        std::vector<std::vector<std::pair<int, int>>> children(1);  // (letter, child) in insertion order.
        std::vector<int> outputs(1, 0);
        for (const Sample& sample : samples) {
            int node = 0;
            for (int a : sample.word) {
                int next = -1;
                for (auto [child_letter, child] : children[node])
                    if (child_letter == a) next = child;
                if (next < 0) {
                    next = static_cast<int>(children.size());
                    children[node].emplace_back(a, next);
                    children.emplace_back();
                    outputs.push_back(0);
                }
                node = next;
            }
            int sign = sample.accepts ? 1 : -1;
            if (outputs[node] == -sign) throw std::runtime_error("a word has conflicting outputs");
            outputs[node] = sign;
        }

        std::vector<bool> informative(children.size());
        for (int node = static_cast<int>(children.size()) - 1; node >= 0; --node) {
            informative[node] = outputs[node] != 0;
            for (auto [_, child] : children[node]) informative[node] = informative[node] || informative[child];
        }

        std::vector<int> tree_nodes{0};
        parent.push_back(-1);
        letter.push_back(-1);
        for (size_t node = 0; node < tree_nodes.size(); ++node)
            for (auto [a, child] : children[tree_nodes[node]])
                if (informative[child]) {
                    parent.push_back(static_cast<int>(node));
                    letter.push_back(a);
                    tree_nodes.push_back(child);
                }
        for (int tree_node : tree_nodes) output.push_back(outputs[tree_node]);
        index_suffixes();
    }

    int size() const { return static_cast<int>(output.size()); }

    void index_suffixes() {
        std::unordered_map<long long, int> suffix_numbers;  // (letter, suffix) -> suffix.
        suffix_occurrences.assign(1, {});
        below.assign(size(), {});
        for (int node = 0; node < size(); ++node) {
            int sign = output[node];
            if (!sign) continue;
            for (int suffix = 0, ancestor = node;; ancestor = parent[ancestor]) {
                suffix_occurrences[suffix].emplace_back(ancestor, sign);
                below[ancestor].emplace_back(suffix, sign);
                if (parent[ancestor] < 0) break;
                long long key = static_cast<long long>(suffix) * alphabet_size + letter[ancestor];
                auto [it, inserted] = suffix_numbers.emplace(key, static_cast<int>(suffix_occurrences.size()));
                if (inserted) suffix_occurrences.emplace_back();
                suffix = it->second;
            }
        }
    }

    // Per node: whether it is apart from `node`.
    std::vector<bool> apart_from(int node) const {
        std::vector<bool> apart(size());
        for (auto [suffix, sign] : below[node])
            for (auto [other, other_sign] : suffix_occurrences[suffix])
                if (other_sign == -sign) apart[other] = true;
        return apart;
    }
};

// CNF of DfaEncoding.py, added directly to a CaDiCaL solver.
class DfaEncoding {
public:
    DfaEncoding(CaDiCaL::Solver& solver, const ObservationTree& tree, int number_of_states, int basis_size,
                const std::vector<std::vector<int>>& mapping_domains)
        : solver_(solver), states_(number_of_states), alphabet_size_(tree.alphabet_size) {
        transitions_.resize(static_cast<size_t>(states_) * alphabet_size_);
        for (auto& transition : transitions_) transition = new_vars(states_);
        outputs_ = new_vars(states_);
        for (const auto& transition : transitions_) exactly_one(transition);
        encode_mapping(mapping_domains, basis_size);
        encode_functional_simulation(tree);
        encode_outputs(tree);
        encode_output_signatures(tree);
        encode_state_activation(basis_size);
    }

    // Assumption for "only the first `bound` states are used"; 0 (none) at the maximum.
    int bound_assumption(int bound) const { return bound < states_ ? off_[bound] : 0; }

    Hypothesis decode(int number_of_states) const {
        Hypothesis hypothesis;
        for (int state = 0; state < number_of_states; ++state) {
            hypothesis.accepting.push_back(solver_.val(outputs_[state]) > 0);
            auto& targets = hypothesis.transitions.emplace_back();
            for (int a = 0; a < alphabet_size_; ++a) {
                const auto& transition = transition_vars(state, a);
                int target = 0;
                while (solver_.val(transition[target]) <= 0) ++target;
                targets.push_back(target);
            }
        }
        return hypothesis;
    }

private:
    CaDiCaL::Solver& solver_;
    int states_, alphabet_size_;
    int num_vars_ = 0;
    std::vector<std::vector<int>> transitions_;  // delta(s, a, t) at [s * alphabet_size + a][t].
    std::vector<int> outputs_;                   // accepting(s).
    std::vector<std::vector<int>> mapping_;      // Per node: map(u, s) per state, 0 outside the domain.
    std::vector<std::vector<std::pair<int, int>>> mapping_items_;  // Per node: (s, map(u, s)).
    std::vector<int> off_;                       // Per state t >= basis size: states t, t+1, ... are unused.

    std::vector<int> new_vars(int count) {
        std::vector<int> vars(count);
        for (int& var : vars) var = ++num_vars_;
        return vars;
    }

    void add(std::initializer_list<int> clause) {
        for (int literal : clause) solver_.add(literal);
        solver_.add(0);
    }

    const std::vector<int>& transition_vars(int state, int a) const {
        return transitions_[static_cast<size_t>(state) * alphabet_size_ + a];
    }

    void exactly_one(const std::vector<int>& literals) {
        for (int literal : literals) solver_.add(literal);
        solver_.add(0);
        int count = static_cast<int>(literals.size());
        if (count <= kPairwiseAtMostOneLimit) {
            for (int i = 0; i < count; ++i)
                for (int j = i + 1; j < count; ++j) add({-literals[i], -literals[j]});
            return;
        }
        std::vector<int> prefix = new_vars(count - 1);  // Sinz's sequential counter.
        add({-literals[0], prefix[0]});
        for (int i = 1; i < count - 1; ++i) {
            add({-prefix[i - 1], prefix[i]});
            add({-literals[i], -prefix[i - 1]});
            add({-literals[i], prefix[i]});
        }
        add({-literals[count - 1], -prefix[count - 2]});
    }

    void encode_mapping(const std::vector<std::vector<int>>& mapping_domains, int basis_size) {
        std::vector<int> used(states_, 0);  // Per state: literal for "some node so far maps to it".
        for (const auto& domain : mapping_domains) {
            std::vector<int> vars = new_vars(static_cast<int>(domain.size()));
            auto& mapping = mapping_.emplace_back(states_, 0);
            auto& items = mapping_items_.emplace_back();
            for (size_t i = 0; i < domain.size(); ++i) {
                mapping[domain[i]] = vars[i];
                items.emplace_back(domain[i], vars[i]);
            }
            exactly_one(vars);
            for (auto [state, literal] : items)
                if (state > basis_size) {
                    if (used[state - 1]) add({-literal, used[state - 1]});
                    else add({-literal});
                }
            for (auto [state, literal] : items) {
                if (state < basis_size) continue;
                int previous = used[state];
                if (!previous) {
                    used[state] = literal;
                    continue;
                }
                int current = used[state] = new_vars(1)[0];
                add({-literal, current});
                add({-previous, current});
                add({-current, previous, literal});
            }
        }
    }

    void encode_functional_simulation(const ObservationTree& tree) {
        for (int node = 1; node < tree.size(); ++node) {
            const auto& mapping = mapping_[node];
            for (auto [state, literal] : mapping_items_[tree.parent[node]]) {
                const auto& transition = transition_vars(state, tree.letter[node]);
                for (int target = 0; target < states_; ++target) {
                    if (mapping[target]) add({-literal, -transition[target], mapping[target]});
                    else add({-literal, -transition[target]});
                }
            }
        }
    }

    void encode_outputs(const ObservationTree& tree) {
        for (int node = 0; node < tree.size(); ++node)
            if (int sign = tree.output[node])
                for (auto [state, literal] : mapping_items_[node]) add({-literal, sign * outputs_[state]});
    }

    void encode_output_signatures(const ObservationTree& tree) {
        for (size_t suffix = 1; suffix < tree.suffix_occurrences.size(); ++suffix) {
            const auto& occurrences = tree.suffix_occurrences[suffix];
            std::vector<bool> accepting(states_), rejecting(states_);
            for (auto [node, sign] : occurrences)
                for (auto [state, _] : mapping_items_[node]) (sign > 0 ? accepting : rejecting)[state] = true;
            std::vector<int> signature(states_, 0);  // sig(s, w) for states with both outputs.
            for (int state = 0; state < states_; ++state)
                if (accepting[state] && rejecting[state]) signature[state] = new_vars(1)[0];
            for (auto [node, sign] : occurrences)
                for (auto [state, literal] : mapping_items_[node])
                    if (signature[state]) add({-literal, sign * signature[state]});
        }
    }

    void encode_state_activation(int basis_size) {
        off_.assign(states_, 0);
        for (int state = basis_size; state < states_; ++state) off_[state] = new_vars(1)[0];
        for (int state = basis_size; state < states_; ++state) {
            if (state + 1 < states_) add({-off_[state], off_[state + 1]});
            for (const auto& mapping : mapping_)
                if (mapping[state]) add({-off_[state], -mapping[state]});
            for (const auto& transition : transitions_) add({-off_[state], -transition[state]});
        }
    }
};

class Learner {
public:
    double solver_time = 0;

    explicit Learner(const ObservationTree& tree) : tree_(tree) {}

    Hypothesis learn() {
        // Basis: repeatedly the first node, breadth-first, that is apart from all basis nodes so far.
        std::vector<bool> candidate(tree_.size(), true);
        for (int node = 0; node < tree_.size(); ++node) {
            if (!candidate[node]) continue;
            std::vector<bool> apart = tree_.apart_from(node);
            basis_.push_back(node);
            compatible_.emplace_back(tree_.size());
            for (int other = 0; other < tree_.size(); ++other) {
                compatible_.back()[other] = !apart[other];
                candidate[other] = candidate[other] && apart[other];
            }
        }
        for (int lower = static_cast<int>(basis_.size());; lower += kExtraStatesPerEncoding + 1) {
            if (auto hypothesis = find_hypothesis(lower, lower + kExtraStatesPerEncoding)) return *hypothesis;
        }
    }

private:
    const ObservationTree& tree_;
    std::vector<int> basis_;
    std::vector<std::vector<bool>> compatible_;  // Per basis state: nodes not apart from its basis node.

    // A DFA with the fewest states in lower..cap, if there is one.
    std::optional<Hypothesis> find_hypothesis(int lower, int cap) {
        auto start = Clock::now();
        int basis_size = static_cast<int>(basis_.size());
        std::vector<std::vector<int>> domains(tree_.size());
        for (int state = 0; state < basis_size; ++state)
            for (int node = 0; node < tree_.size(); ++node)
                if (compatible_[state][node]) domains[node].push_back(state);
        std::vector<bool> in_basis(tree_.size());
        for (int node : basis_) in_basis[node] = true;
        for (int node = 0; node < tree_.size(); ++node)
            if (!in_basis[node])
                for (int state = basis_size; state < cap; ++state) domains[node].push_back(state);

        CaDiCaL::Solver solver;
        solver.set("stabilizeonly", 1);
        DfaEncoding encoding(solver, tree_, cap, basis_size, domains);
        std::optional<Hypothesis> hypothesis;
        for (int bound = lower; bound <= cap && !hypothesis; ++bound) {
            if (int assumption = encoding.bound_assumption(bound)) solver.assume(assumption);
            int result = solver.solve();
            if (result != 10 && result != 20) throw std::runtime_error("SAT solver failed");
            if (result == 10) hypothesis = encoding.decode(bound);
        }
        solver_time += seconds_since(start);
        return hypothesis;
    }
};

std::vector<Sample> read_abbadingo(const std::string& path, int& alphabet_size) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::string line;
    std::getline(in, line);
    int sample_count;
    std::istringstream(line) >> sample_count >> alphabet_size;
    std::vector<Sample> samples;
    while (std::getline(in, line)) {
        std::istringstream values(line);
        int label, length;
        if (!(values >> label >> length) || (label != 0 && label != 1)) continue;
        Sample& sample = samples.emplace_back(Sample{std::vector<int>(length), label == 1});
        for (int& a : sample.word) values >> a;
    }
    return samples;
}

}  // namespace

int main(int argc, char** argv) {
    std::string usage = "usage: lsharp_square FILE [--dfa]\n";
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--dfa")) {
        std::cerr << usage;
        return 2;
    }
    try {
        auto start = Clock::now();
        int alphabet_size;
        std::vector<Sample> samples = read_abbadingo(argv[1], alphabet_size);
        ObservationTree tree(alphabet_size, samples);
        Learner learner(tree);
        Hypothesis dfa = learner.learn();

        bool agrees = true;
        for (const Sample& sample : samples) {
            int state = 0;
            for (int a : sample.word) state = dfa.transitions[state][a];
            agrees = agrees && dfa.accepting[state] == sample.accepts;
        }
        std::string name = argv[1];
        name = name.substr(name.find_last_of('/') + 1);
        std::printf("%s,%zu,%s,%.4f,%.4f\n", name.c_str(), dfa.accepting.size(), agrees ? "True" : "False",
                    learner.solver_time, seconds_since(start));
        if (argc == 3)
            for (size_t state = 0; state < dfa.accepting.size(); ++state) {
                std::printf("%d", static_cast<int>(dfa.accepting[state]));
                for (int target : dfa.transitions[state]) std::printf(" %d", target);
                std::printf("\n");
            }
        return agrees ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 2;
    }
}
