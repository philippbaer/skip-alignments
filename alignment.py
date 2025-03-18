from processtree import *
from typing import List, Optional, Any, Set
from enum import Enum
import queue
from functools import total_ordering
from collections import defaultdict
import time

level_incentive = 0

@total_ordering
class NodeState(Enum):
    CLOSED = 1
    ACTIVE = 2
    ENABLED = 3
    FUTURE = 4

    def __lt__(self, other):
        if not isinstance(other, NodeState):
            raise ValueError("Expected Enum value.")
        return self.value < other.value

@total_ordering
class State(object):
    """
    State in the search space for skip alignments in normal form. It represents the tuple (sigma',delta) of remaining activities to be aligned and the current alignment. delts is guaranteed to be normal formed.
    """
    def __init__(self, tree:ProcessTree, trace:List[str], mapper:"Mapper") -> None:
        self.tree = tree
        self.mapper = mapper
        self.state = [NodeState.ENABLED] + [NodeState.CLOSED]*(mapper.size()-1)
        self.trace = trace
        self.path = []
        self.acc_costs = 0
        self.already_fired = []
        self.and_stops = defaultdict(list) # indices do not respect unfolding
        self.and_starts = defaultdict(list) # indices do not respect unfolding
    
    @staticmethod
    def initial_state(tree:ProcessTree, trace:List[str], mapper:"Mapper"):
        """
        Creates the initial state for a trace and a process tree.
        """
        return State(tree, trace, mapper)
    
    def is_final(self):
        """
        Returns true iff the current state is a goal state.
        """
        return self.state == [NodeState.CLOSED]*self.mapper.size() and len(self.trace) == 0

    def successors2(self, log_move_costs:List[int], time_bound:int):
        """
        Returns a list of successor states reachable from the current state (<a1, ..., an>,delta) by inspecting the next activity a1.

        The successor states are computed by either
        - filling up with skip moves to complete the execution of the tree if n==0, or
        - performing a synchronous move on a1 possibly requiring some skip moves before, or
        - creating a log move on a1.

        Details can be found in the paper section 4.4.
        """
        # idea: for every remaining event, try to directly go there
        if self.is_init() and len(self.trace) == 0:
            # trying to align empty trace, i.e., entire trace before wasn't fitting
            new_state = self.copy()
            self.shortest_execution(self.tree, new_state)
            return [new_state]
        if len(self.trace) == 0:
            # before we aligned, but now trace is done
            # first ensure that the last sync move is covered

            leaf_idxs = [self.mapper.node_to_index(leaf) for leaf in self.tree.get_leafs()]
            current_activity = None
            for i, idx in enumerate(leaf_idxs):
                if self.state[idx] == NodeState.ACTIVE:
                    current_activity = self.tree.get_leafs()[i]
            new_state = self.finish_subtree(self.tree, current_activity)
            return [new_state]
        # still an activity left, try to align it
        successor_states = self.shortest_path_to_activate(self.trace[0], time_bound)
        # or just skip over the next move
        new_state = self.copy()
        new_state.path += [(self.trace[0], ">>")]
        new_state.trace = new_state.trace[1:]
        new_state.acc_costs += log_move_costs[-len(self.trace)]
        return successor_states + [new_state]
    
    def shortest_path_to_activate(self, activity:str, time_bound) -> Optional[List[Any]]:
        if activity not in self.tree.get_leaf_labels():
            return [] # activity does not exist
        new_states = []
        if self.is_init():
            # sure there exists a path
            assert len(self.already_fired) == 0
            for leaf in [l for l in self.tree.get_leafs() if l.name == activity and isinstance(l, Activity)]:
                # these leafs would match and a path to each does exist
                new_state = self.get_shortest_path_down(self.tree, leaf)
                assert new_state is not None
                new_states.append(new_state)
                new_state.trace = new_state.trace[1:]
        else:
            # check, if from current activity exists a path
            # NOTE: still need to fire the last activity
            leaf_idxs = [self.mapper.node_to_index(leaf) for leaf in self.tree.get_leafs()]
            current_activity = None
            for i, idx in enumerate(leaf_idxs):
                if self.state[idx] == NodeState.ACTIVE:
                    current_activity = self.tree.get_leafs()[i]
            assert current_activity is not None

            for leaf in [l for l in self.tree.get_leafs() if l.name == activity and isinstance(l, Activity)]:
                # these leafs would match but not neccessarily a path does exist
                if self.is_execution_order_possible(current_activity, leaf):
                    lcas = self.get_generalized_lca(current_activity, leaf)
                    for lca, belongs_to_a_higher_and in lcas:
                        #print("Ispecting LCA:", lca)
                        if time.process_time() > time_bound:
                            return []
                        new_state = self.complete_running_subtree(current_activity, leaf, lca if not belongs_to_a_higher_and else And(None,[]))

                        if isinstance(lca, Sequence):
                            for c in lca.children:
                                # we know that the leaf is more to the right, i.e., future yet
                                if new_state.state[self.mapper.node_to_index(c)] == NodeState.CLOSED:
                                    continue
                                assert new_state.state[self.mapper.node_to_index(c)] == NodeState.FUTURE
                                if c.contains_tree(leaf):
                                    # go in
                                    new_state = self.get_shortest_path_down(c, leaf, new_state)
                                    break
                                else:
                                    self.shortest_execution(c, new_state)
                        elif isinstance(lca, Xor):
                            # this is impossible
                            raise ValueError("Cannot re-deepen an Xor node:", lca, current_activity, leaf)
                        elif isinstance(lca, And):
                            for c in lca.children:
                                # we know that the leaf is not executed yet
                                if new_state.state[self.mapper.node_to_index(c)] == NodeState.CLOSED:
                                    continue
                                assert new_state.state[self.mapper.node_to_index(c)] == NodeState.ENABLED or NodeState.ACTIVE
                                if c.contains_tree(leaf):
                                    # go in
                                    start_node = leaf
                                    while new_state.state[self.mapper.node_to_index(start_node)] != NodeState.ENABLED and new_state.state[self.mapper.node_to_index(start_node)] != NodeState.ACTIVE:
                                        start_node = start_node.parent
                                    assert start_node is not None
                                    if new_state.state[self.mapper.node_to_index(start_node)] == NodeState.ENABLED:
                                        # we can directly go down
                                        new_state = self.get_shortest_path_down(start_node, leaf, new_state)
                                    else:
                                        # we might need to complete the last subtrees (ACTIVE)
                                        assert isinstance(start_node, Sequence) or isinstance(start_node, Loop)
                                        # close the current activity
                                        new_state.state[self.mapper.node_to_index(current_activity)] = NodeState.CLOSED
                                        if isinstance(start_node, Sequence):
                                            for c2 in start_node.children:
                                                if new_state.state[self.mapper.node_to_index(c2)] == NodeState.CLOSED:
                                                    continue
                                                if new_state.state[self.mapper.node_to_index(c2)] == NodeState.ACTIVE:
                                                    #print("Finishing the other branch", c2)
                                                    self.complete_partial_tree(c2, new_state)
                                                if new_state.state[self.mapper.node_to_index(c2)] == NodeState.FUTURE:
                                                    # maybe we need to go in
                                                    if c2.contains_tree(leaf):
                                                        # go in
                                                        new_state.state[self.mapper.node_to_index(c2)] == NodeState.ENABLED
                                                        new_state = self.get_shortest_path_down(c2, leaf, new_state)
                                                        break
                                                    else:
                                                        # first do another subtree
                                                        self.shortest_execution(c2, new_state)
                                        elif isinstance(start_node, Loop):
                                            # maybe a redo execution is needed
                                            # and maybe we need to finish the old execution
                                            last_was_do = False
                                            branch = None
                                            for i in range(len(new_state.already_fired)):
                                                if branch is not None:
                                                    break
                                                if start_node.children[0].contains_tree(new_state.already_fired[-(i+1)]):
                                                    # last was do
                                                    last_was_do = True
                                                    branch = start_node.children[0]
                                                    break
                                                for c3 in start_node.children[1:]:
                                                    if c3.contains_tree(new_state.already_fired[-(i+1)]):
                                                        # last was redo
                                                        last_was_do = False
                                                        branch = c3
                                                        break
                                            next_is_do = True
                                            next_branch = start_node.children[0]
                                            for c3 in start_node.children[1:]:
                                                if c3.contains_tree(leaf):
                                                    next_is_do = False
                                                    next_branch = c3
                                                    break
                                            # finish do or redo execution
                                            if branch is not None:
                                                self.complete_partial_tree(branch, new_state)
                                            if last_was_do:
                                                # we finished the do already
                                                if next_is_do:
                                                    # we need to run a redo first, then go into do
                                                    new_states2:List["State"] = []
                                                    min_idx = -1
                                                    for c3 in start_node.children[1:]:
                                                        new_state2 = new_state.copy()
                                                        self.shortest_execution(c3, new_state2)
                                                        new_states2.append(new_state2)
                                                        if min_idx == -1 or new_states2[min_idx].acc_costs > new_state2.acc_costs:
                                                            min_idx = len(new_states2)-1
                                                    new_state = new_states2[min_idx]
                                                    new_state = self.get_shortest_path_down(start_node.children[0], leaf, new_state)
                                                else:
                                                    # directly go into redo
                                                    new_state = self.get_shortest_path_down(next_branch, leaf, new_state)
                                            else:
                                                # we finished the redo already
                                                if next_is_do:
                                                    # we can go into do directly
                                                    new_state = self.get_shortest_path_down(next_branch, leaf, new_state)
                                                else:
                                                    # again a redo is needed
                                                    self.complete_partial_tree(start_node.children[0], new_state)
                                                    new_state = self.get_shortest_path_down(next_branch, leaf, new_state)
                                    break
                        elif isinstance(lca, Loop):
                            current_activity_belonging_to_loop = current_activity
                            found_last_activity = False
                            for i in range(len(new_state.already_fired)):
                                if found_last_activity:
                                    break
                                for loop_c in lca.children:
                                    if loop_c.contains_tree(new_state.already_fired[len(new_state.already_fired)-i-1]):
                                        found_last_activity = True
                                        current_activity_belonging_to_loop = new_state.already_fired[len(new_state.already_fired)-i-1]
                                        break
                            if belongs_to_a_higher_and:
                                # the loop is triggered on the path from the real LCA And to an activity, i.e., we need to finish the execution of the loop first
                                new_state = new_state.complete_running_subtree(current_activity_belonging_to_loop, leaf, lca)

                            if lca.children[0].contains_tree(current_activity_belonging_to_loop):
                                # we executed the do part
                                if lca.children[0].contains_tree(leaf):
                                    # again do part
                                    new_states2:List["State"] = []
                                    min_idx = -1
                                    for c in lca.children[1:]:
                                        new_state2 = new_state.copy()
                                        self.shortest_execution(c, new_state2)
                                        new_states2.append(new_state2)
                                        if min_idx == -1 or new_states2[min_idx].acc_costs > new_state2.acc_costs:
                                            min_idx = len(new_states2)-1
                                    new_state = new_states2[min_idx]
                                    new_state = self.get_shortest_path_down(lca.children[0], leaf, new_state)
                                else:
                                    # now it is a redo part
                                    for c in lca.children[1:]:
                                        if c.contains_tree(leaf):
                                            new_state = self.get_shortest_path_down(c, leaf, new_state)
                                            break
                            else:
                                # we executed the redo part
                                if lca.children[0].contains_tree(leaf):
                                    # now it is do part
                                    new_state = self.get_shortest_path_down(lca.children[0], leaf, new_state)
                                else:
                                    # now it is again redo part
                                    self.shortest_execution(lca.children[0], new_state)
                                    for c in lca.children[1:]:
                                        if c.contains_tree(leaf):
                                            new_state = self.get_shortest_path_down(c, leaf, new_state)
                        else:
                            raise ValueError("Turning at LCA did not work at:", lca, current_activity, leaf)
                        new_states.append(new_state)
                        new_state.trace = new_state.trace[1:]
                else:
                    pass
        return new_states


    def complete_running_subtree(self, act_running:ProcessTree, act_new:ProcessTree,lca:ProcessTree) -> "State":
        # NOTE: This does also fire the current activity
        
        if isinstance(lca, And):
            # we do not really need to complete the entire tree first
            new_state = self.copy()
            new_state.state[self.mapper.node_to_index(act_running)] = NodeState.CLOSED
            # maybe we want to propagate up what can certainly be closed
            return new_state

        nodes_to_lca = [act_running] # [act_running, ..., child of lca, NOT lca]
        current_node = act_running.parent
        while current_node != lca:
            nodes_to_lca.append(current_node)
            current_node = current_node.parent
        
        return self.finish_subtree(nodes_to_lca[-1], act_running) # finish up the entire node

    def finish_subtree(self, root:ProcessTree, act_running:ProcessTree):
        # fully completes the execution of that node
        new_state = self.copy()
        new_state.state[self.mapper.node_to_index(act_running)] = NodeState.CLOSED

        # finish the subtree execution of the running activity until lca
        current_root = act_running.parent
        last_node = act_running
        while current_root != root.parent:
            self.execute_tree(current_root, new_state, last_node)
            last_node = current_root
            current_root = current_root.parent
        
        return new_state
        

    
    def execute_tree(self, root:ProcessTree, state:"State", last_node:ProcessTree) -> None:
        # finishes the execution of a subtree where some parts might already be executed
        assert state.state[self.mapper.node_to_index(root)] == NodeState.ACTIVE

        if isinstance(root, LeafNode):
            raise ValueError("No leaf activity expected to happen here")
        if isinstance(root, Sequence):
            for c in root.children:
                if state.state[self.mapper.node_to_index(c)] == NodeState.FUTURE:
                    # finish execution of a yet uninspected trees
                    self.shortest_execution(c, state)
            state.state[self.mapper.node_to_index(root)] = NodeState.CLOSED
            return
        if isinstance(root, Xor):
            # nothing to do, execution already finished
            state.state[self.mapper.node_to_index(root)] = NodeState.CLOSED
            return
        if isinstance(root, And):
            for c in root.children:
                if state.state[self.mapper.node_to_index(c)] == NodeState.ENABLED:
                    # finish yet uninspected trees, do it fully
                    self.shortest_execution(c, state)
                elif state.state[self.mapper.node_to_index(c)] == NodeState.ACTIVE:
                    # the subtree was already inspected partially
                    self.complete_partial_tree(c, state)
            state.state[self.mapper.node_to_index(root)] = NodeState.CLOSED
            state.and_stops[root].append(state.get_last_executed_event_idx(root))
            return
        if isinstance(root, Loop):
            if last_node == root.children[0]:
                # do was last, nothing to do
                state.state[self.mapper.node_to_index(root)] = NodeState.CLOSED
                return
            else:
                # redo was last
                self.shortest_execution(root.children[0], state)
                state.state[self.mapper.node_to_index(root)] = NodeState.CLOSED
                return
        raise ValueError("No node was found to be handeled in finishing the subtree:", root)

    def complete_partial_tree(self, tree:ProcessTree, state:"State"):
        # does complete a branch of an And node. When called, tree is NEVER future. The last running activity is already set to CLOSED.

        if state.state[self.mapper.node_to_index(tree)] == NodeState.CLOSED:
            # what ever the subtree is, it was already executed, i.e., nothing to do
            return

        if isinstance(tree, LeafNode):
            if state.state[self.mapper.node_to_index(tree)] == NodeState.ENABLED:
                self.shortest_execution(tree, state)
                state.state[self.mapper.node_to_index(tree)] = NodeState.CLOSED
            elif state.state[self.mapper.node_to_index(tree)] == NodeState.ACTIVE:
                raise ValueError("We expected the last running activity to be closed:", tree)
        elif isinstance(tree, Sequence):
            if state.state[self.mapper.node_to_index(tree)] == NodeState.ENABLED:
                # the subtree was never entered
                self.shortest_execution(tree, state)
                state.state[self.mapper.node_to_index(tree)] = NodeState.CLOSED
            elif state.state[self.mapper.node_to_index(tree)] == NodeState.ACTIVE:
                # is already running
                for c in tree.children:
                    if state.state[self.mapper.node_to_index(c)] == NodeState.CLOSED:
                        # nothing to do
                        pass
                    elif state.state[self.mapper.node_to_index(c)] == NodeState.ENABLED:
                        raise ValueError("A child of a running Sequence is closed, future, or active")
                    elif state.state[self.mapper.node_to_index(c)] == NodeState.ACTIVE:
                        # still running
                        self.complete_partial_tree(c, state)
                        state.state[self.mapper.node_to_index(c)] = NodeState.CLOSED
                    elif state.state[self.mapper.node_to_index(c)] == NodeState.FUTURE:
                        # not yet explored
                        self.shortest_execution(c, state)
                        state.state[self.mapper.node_to_index(c)] = NodeState.CLOSED
                state.state[self.mapper.node_to_index(tree)] = NodeState.CLOSED
            else:
                raise ValueError("Unexpected state of tree:", tree, state.state[self.mapper.node_to_index(tree)])
        elif isinstance(tree, Xor):
            if state.state[self.mapper.node_to_index(tree)] == NodeState.ACTIVE:
                # already running
                running_child = None
                for c in tree.children:
                    if state.state[self.mapper.node_to_index(c)] == NodeState.ACTIVE:
                        running_child = c
                if running_child is not None:
                    # still running
                    self.complete_partial_tree(running_child, state)
                    state.state[self.mapper.node_to_index(tree)] = NodeState.CLOSED
                else:
                    # everything below was already executed
                    state.state[self.mapper.node_to_index(tree)] = NodeState.CLOSED
            else:
                raise ValueError("Unexpected state of tree:", tree, state.state[self.mapper.node_to_index(tree)])
        elif isinstance(tree, And):
            if state.state[self.mapper.node_to_index(tree)] == NodeState.ENABLED:
                self.shortest_execution(tree, state)
                state.state[self.mapper.node_to_index(tree)] = NodeState.CLOSED
                state.and_stops[tree].append(state.get_last_executed_event_idx(tree))
            elif state.state[self.mapper.node_to_index(tree)] == NodeState.ACTIVE:
                for c in tree.children:
                    if state.state[self.mapper.node_to_index(c)] == NodeState.CLOSED:
                        # child already done
                        pass
                    elif state.state[self.mapper.node_to_index(c)] == NodeState.ENABLED:
                        #print("Finishing execution of child", c)
                        self.shortest_execution(c, state)
                        #print(agn2, lfs2)
                        state.state[self.mapper.node_to_index(c)] = NodeState.CLOSED
                    elif state.state[self.mapper.node_to_index(c)] == NodeState.ACTIVE:
                        self.complete_partial_tree(c, state)
                        state.state[self.mapper.node_to_index(c)] = NodeState.CLOSED
                    else:
                        raise ValueError("Enexpected state of child:", c, state.state[self.mapper.node_to_index(c)])
                state.state[self.mapper.node_to_index(tree)] = NodeState.CLOSED
                state.and_stops[tree].append(state.get_last_executed_event_idx(tree))
            else:
                raise ValueError("Enexpected state of tree:", tree, state.state[self.mapper.node_to_index(tree)])
        elif isinstance(tree, Loop):
            if state.state[self.mapper.node_to_index(tree)] == NodeState.ENABLED:
                self.shortest_execution(tree, state)
                state.state[self.mapper.node_to_index(tree)] = NodeState.CLOSED
            elif state.state[self.mapper.node_to_index(tree)] == NodeState.ACTIVE:
                if state.state[self.mapper.node_to_index(tree.children[0])] == NodeState.ACTIVE:
                    # do is running
                    self.complete_partial_tree(tree.children[0], state)
                    state.state[self.mapper.node_to_index(tree.children[0])] = NodeState.CLOSED
                elif state.state[self.mapper.node_to_index(tree.children[0])] == NodeState.ENABLED:
                    self.shortest_execution(tree.children[0], state)
                    state.state[self.mapper.node_to_index(tree.children[0])] = NodeState.CLOSED
                else:
                    # maybe a redo is still running
                    need_to_find = True
                    for c in tree.children:
                        if state.state[self.mapper.node_to_index(c)] == NodeState.ACTIVE:
                            self.complete_partial_tree(c, state)
                            state.state[self.mapper.node_to_index(c)] = NodeState.CLOSED
                            # run do again
                            self.shortest_execution(tree.children[0], state)
                            need_to_find = False
                            break
                    if need_to_find:
                        # we need to check which execution was last, whether it is finished and what branch to take next
                        last_was_do = False
                        branch = None
                        for i in range(len(state.already_fired)):
                            if branch is not None:
                                break
                            if tree.children[0].contains_tree(state.already_fired[-(i+1)]):
                                # last was do
                                last_was_do = True
                                branch = tree.children[0]
                                break
                            for c3 in tree.children[1:]:
                                if c3.contains_tree(state.already_fired[-(i+1)]):
                                    # last was redo
                                    last_was_do = False
                                    branch = c3
                                    break
                        # finish do or redo execution
                        if branch is not None:
                            self.complete_partial_tree(branch, state)
                        if last_was_do:
                            # we finished the do already
                            pass
                        else:
                            # we finished the redo already
                            self.shortest_execution(tree.children[0], state)
                    # a ENABLED cannot happen
                state.state[self.mapper.node_to_index(tree)] = NodeState.CLOSED
            else:
                raise ValueError("Unexpected state of tree:", tree, state.state[self.mapper.node_to_index(tree)])
        else:
            raise ValueError("Unexpected type of node", tree)
        return

    def get_generalized_lca(self, act1:ProcessTree, act2:ProcessTree) -> List["State"]:
        lca = self.tree.get_lca(act1, act2)
        assert lca is not None

        if isinstance(lca, LeafNode):
            # same node again
            return [(x,False) for x in self.get_all_loops_before(lca)]
        elif isinstance(lca, Sequence):
            # current activity has to come first or a Loop was before
            for c in lca.children:
                if c.contains_tree(act1):
                    return [(lca,False)] + [(x,False) for x in self.get_all_loops_before(lca.parent)]
                if c.contains_tree(act2):
                    return [(x,False) for x in self.get_all_loops_before(lca)]
        elif isinstance(lca, Xor):
            # only possible, if a Loop node was executed before
            return [(x,False) for x in self.get_all_loops_before(lca)]
        elif isinstance(lca, And):
            # either the leaf is still awaiting execution of the And or a Loop was before
            lcas = []
            for c in lca.children:
                if c.contains_tree(act2):
                    current_node = act2
                    while current_node != lca: # NOTE: Enough if on the path from the And to the node is an ENABLED node
                        if self.state[self.mapper.node_to_index(current_node)] == NodeState.ENABLED or self.state[self.mapper.node_to_index(current_node)] == NodeState.FUTURE:
                            lcas.append((lca,False)) # from the lca there exists a path of execution that can fire or continue without needing to loop
                        if (self.state[self.mapper.node_to_index(current_node)] == NodeState.ACTIVE and isinstance(current_node, Loop)):
                            lcas.append((current_node,True)) # on the path AND --> act2 there is a loop that can iterate
                        current_node = current_node.parent
                    # still possible as another branch was chosen
            return lcas + [(x,False) for x in self.get_all_loops_before(lca)]
        elif isinstance(lca, Loop):
            # a loop can always repeat
            return [(lca,False)] + [(x,False) for x in self.get_all_loops_before(lca.parent)]
    
    def get_next_loop(self, node:ProcessTree):
        if isinstance(node, Loop):
            return node
        assert node.parent is not None
        return self.get_next_loop(node.parent)
    
    def get_all_loops_before(self, node:ProcessTree):
        if node is None:
            return []
        if isinstance(node, Loop):
            return [node] + self.get_all_loops_before(node.parent)
        else:
            return self.get_all_loops_before(node.parent)

    def is_execution_order_possible(self, act1:ProcessTree, act2:ProcessTree):
        lca = self.tree.get_lca(act1, act2)
        assert lca is not None

        if isinstance(lca, LeafNode):
            # same node again
            return self.check_if_loop_before(lca)
        elif isinstance(lca, Sequence):
            # current activity has to come first or a Loop was before
            for c in lca.children:
                if c.contains_tree(act1):
                    return True
                if c.contains_tree(act2):
                    return self.check_if_loop_before(lca)
        elif isinstance(lca, Xor):
            # only possible, if a Loop node was executed before
            return self.check_if_loop_before(lca)
        elif isinstance(lca, And):
            # either the leaf is still awaiting execution of the And or a Loop was before
            for c in lca.children:
                if c.contains_tree(act2):
                    current_node = act2
                    while current_node != lca: # NOTE: Enough if on the path from the And to the node is an ENABLED node
                        if self.state[self.mapper.node_to_index(current_node)] == NodeState.ENABLED or self.state[self.mapper.node_to_index(current_node)] == NodeState.FUTURE or (self.state[self.mapper.node_to_index(current_node)] == NodeState.ACTIVE and isinstance(current_node, Loop)):
                            return True 
                        current_node = current_node.parent
                    # still possible as another branch was chosen
            return self.check_if_loop_before(lca)
        elif isinstance(lca, Loop):
            # a loop can always repeat
            return True
        else:
            raise ValueError("Did not expect the LCA to be a non-inner node:", lca, act1, act2)

    def check_if_loop_before(self, node:ProcessTree):
        while node is not None:
            if isinstance(node, Loop):
                return True
            node = node.parent
        return False

    def get_shortest_path_down(self, start_node:ProcessTree, leaf_node:ProcessTree, new_state:"State"=None):
        if start_node == leaf_node:
            # only activity node
            if new_state is None:
                new_state = self.copy()
            new_state.state[self.mapper.node_to_index(leaf_node)] = NodeState.ACTIVE
            new_state.acc_costs += leaf_node.sync_move_cost
            new_state.path += [(leaf_node.name, leaf_node)]
            new_state.already_fired += [leaf_node]
            return new_state
        reverse_node_path = [] # [NOT start_node, start_node+1, ..., leaf_node]
        reverse_node_path.append(leaf_node)
        current_node = leaf_node.parent
        while current_node != start_node:
            reverse_node_path.append(current_node)
            current_node = current_node.parent
        reverse_node_path.reverse()

        # go down
        if new_state is None:
            new_state = self.copy()
        current_node = start_node
        while len(reverse_node_path) > 0:
            next_node = reverse_node_path[0]
            reverse_node_path = reverse_node_path[1:]
            if isinstance(current_node, Sequence):
                for i in range(len(current_node.children)):
                    # only do the subtree executions before the current one
                    if current_node.children[i] != next_node:
                        self.shortest_execution(current_node.children[i], new_state)
                        # update state
                    else:
                        # set others to future
                        new_state.state[self.mapper.node_to_index(current_node)] = NodeState.ACTIVE
                        for j in range(i+1, len(current_node.children)):
                            new_state.state[self.mapper.node_to_index(current_node.children[j])] = NodeState.FUTURE
                        break
            elif isinstance(current_node, Xor):
                # no update needed, only choose the right path
                for c in current_node.children:
                    assert new_state.state[self.mapper.node_to_index(c)] == NodeState.CLOSED
                new_state.state[self.mapper.node_to_index(current_node)] = NodeState.ACTIVE
            elif isinstance(current_node, And):
                # enable all, but only execute the needed one
                new_state.and_starts[current_node].append(len(new_state.path))
                for i in range(len(current_node.children)):
                    # only do enablement
                    if current_node.children[i] != next_node:
                        new_state.state[self.mapper.node_to_index(current_node.children[i])] = NodeState.ENABLED
                    else:
                        # set real one to active
                        new_state.state[self.mapper.node_to_index(current_node)] = NodeState.ACTIVE
            elif isinstance(current_node, Loop):
                # which branch
                if current_node.children.index(next_node) == 0:
                    # only go into the do part
                    new_state.state[self.mapper.node_to_index(current_node)] = NodeState.ACTIVE
                    for j in range(1,len(current_node.children)):
                        assert new_state.state[self.mapper.node_to_index(current_node.children[j])] == NodeState.CLOSED
                else:
                    # first a do, then go into redo
                    self.shortest_execution(current_node.children[0], new_state)
                    # update state
                    new_state.state[self.mapper.node_to_index(current_node)] = NodeState.ACTIVE
            elif isinstance(current_node, Activity):
                # the final destination
                new_state.state[self.mapper.node_to_index(current_node)] = NodeState.ACTIVE
                # do not yet do the execution

            current_node = next_node
        new_state.state[self.mapper.node_to_index(leaf_node)] = NodeState.ACTIVE
        new_state.acc_costs += leaf_node.sync_move_cost
        new_state.path += [(leaf_node.name, leaf_node)]
        new_state.already_fired += [leaf_node]
        return new_state

    def shortest_execution(self, tree:ProcessTree, new_state:"State"):
        agn, lfs = self._shortest_execution(tree, new_state)

        cost_skip, only_taus = tree.get_cheapest_execution(level_incentive)
        if only_taus:
            # there exists a path only of taus that we want to take
            new_state.already_fired += lfs
            new_state.path += [('>>', TauPath(tree))]
            new_state.acc_costs += 0 #sum(leaf.skip_cost for leaf in lfs)
        else:
            # skipping is cheaper
            new_state.already_fired += lfs
            new_state.path += [('>>', Skip(tree, cost_skip))]
            new_state.acc_costs += cost_skip

    def _shortest_execution(self, tree:ProcessTree, new_state:"State"):
        # assumes that the current node is not executed at all!
        if isinstance(tree, LeafNode):
            new_state.state[self.mapper.node_to_index(tree)] = NodeState.CLOSED
            return [('>>', tree)], [tree]
        agn_path = []
        fired_leafs = []
        if isinstance(tree, Sequence):
            for c in tree.children:
                a, f = self._shortest_execution(c, new_state)
                agn_path += a
                fired_leafs += f
            new_state.state[self.mapper.node_to_index(tree)] = NodeState.CLOSED
            return agn_path, fired_leafs
        elif isinstance(tree, Xor):
            new_states = [new_state.copy() for _ in tree.children]
            res = [self._shortest_execution(c, new_states[i]) for i, c in enumerate(tree.children)]
            costs = [sum(leaf.skip_cost for leaf in execs[1]) for execs in res]
            min_idx = costs.index(min(costs))
            new_state.state = new_states[min_idx].state
            new_state.state[self.mapper.node_to_index(tree)] = NodeState.CLOSED
            return res[min_idx]
        elif isinstance(tree, And):
            for c in tree.children:
                a, f = self._shortest_execution(c, new_state)
                agn_path += a
                fired_leafs += f
            new_state.state[self.mapper.node_to_index(tree)] = NodeState.CLOSED
            return agn_path, fired_leafs
        elif isinstance(tree, Loop):
            a, f = self._shortest_execution(tree.children[0], new_state)
            agn_path += a
            fired_leafs += f
            new_state.state[self.mapper.node_to_index(tree)] = NodeState.CLOSED
            return agn_path, fired_leafs
        else:
            raise("Unexpected node type in trying to do complete subtree execution:", tree)

    def get_last_executed_event_idx(self, tree:ProcessTree) -> int:
        # returns the index in the folded path at which the most recent activity execution belonging to tree arised
        for i in range(len(self.path)-1, -1, -1):
            model_move = self.path[i][1]
            if model_move == '>>':
                continue
            elif isinstance(model_move, Skip):
                model_move = model_move.node
            elif isinstance(model_move, TauPath):
                model_move = model_move.node
            
            if tree.contains_tree(model_move):
                return i
        raise ValueError("Did not expect the finished node ")

    def unfold(self):
        assert self.is_final()
        indizes = []
        paths = []
        for i, (_, model_move) in enumerate(self.path):
            if isinstance(model_move, TauPath):
                paths.append(model_move.unfold())
                indizes.append(i)
        if len(indizes) == 0:
            # there is no TauPath
            return [self]
        res = []
        for path in list(itertools.product(*paths)):
            new_state = self.copy()
            for i, elem in enumerate(path):
                # elem = [a,b] [c,d], i.e., we need to flat afterwards
                new_state.path[indizes[i]] = [('>>', tau) for tau in elem]
            new_list = []
            path_pointer = 0
            for j, elem in enumerate(new_state.path):
                if isinstance(elem, tuple):
                    new_list.append(elem)
                    path_pointer += 1
                else:
                    new_list += elem
                    for node, idxs in new_state.and_starts.items():
                        new_idxs = []
                        for idx in idxs:
                            if idx <= path_pointer:
                                new_idxs.append(idx)
                            else:
                                # update the index to respect the unfolded length
                                new_idxs.append(idx + len(elem) - 1)
                        new_state.and_starts[node] = new_idxs
                    for node, idxs in new_state.and_stops.items():
                        new_idxs = []
                        for idx in idxs:
                            if idx <= path_pointer:
                                new_idxs.append(idx)
                            else:
                                # update the index to respect the unfolded length
                                new_idxs.append(idx + len(elem) - 1)
                        new_state.and_stops[node] = new_idxs
                    path_pointer += len(elem)
            new_state.path = new_list
            res.append(new_state)

        return res

    def is_init(self):
        for i in self.state[1:]:
            if i != NodeState.CLOSED:
                return False
        return self.state[0] == NodeState.ENABLED

    def copy(self):
        state = State(self.tree, self.trace, self.mapper)
        for i, v in enumerate(self.state):
            state.state[i] = v
        state.path = []
        for i, v in enumerate(self.path):
            state.path.append(v)
        state.acc_costs = self.acc_costs
        state.already_fired = [f for f in self.already_fired]
        for k,v in self.and_starts.items():
            state.and_starts[k] = [w for w in v]
        for k,v in self.and_stops.items():
            state.and_stops[k] = [w for w in v]
        return state
    
    def heuristic(self):
        """
        By now, no heuristic is needed
        """
        return 0
    
    def costs(self):
        return self.acc_costs + self.heuristic()
    
    def __lt__(self, other):
        if not isinstance(other, State):
            return NotImplemented
        return self.costs() < other.costs() or (self.costs() == other.costs() and self.state < other.state) or (self.costs() == other.costs() and self.state == other.state and self.trace < other.trace)
    
    def __str__(self) -> str:
        return "[ State: " + self.state.__str__() + ", Trace: " + self.trace.__str__() + ", Path: " + self.path.__str__() + ", Cost: " + str(self.acc_costs) + ", AND starts: " + ",".join(["( " + k.id + ", " + v.__str__() + " )" for k,v in self.and_starts.items()]) + ", AND stops: " + ",".join(["( " + k.id + ", " + v.__str__() + " )" for k,v in self.and_stops.items()]) + " ]"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def matches_without_cost(self, other:"State"):
        return self.state == other.state and self.trace == other.trace

class Aligner(object):
    """
    Wrapper for the A* algorithm to compute optimal akip alignments in normal form.
    """

    def __init__(self, tree:ProcessTree) -> None:
        self.tree = tree
        self.mapper = Mapper(self.tree)

    @staticmethod
    def set_level_incentive(l):
        global level_incentive
        level_incentive = l

    def max_tau_cost(self, node:ProcessTree):
        if isinstance(node, LeafNode):
            if isinstance(node, Tau):
                return node.skip_cost
            else:
                return 0
        else:
            return max([self.max_tau_cost(c) for c in node.children])
    
    def min_model_move_cost(self, node:ProcessTree):
        if isinstance(node, LeafNode):
            if isinstance(node, Tau):
                return 9223372036854775807
            else:
                return node.skip_cost
        else:
            return min([self.min_model_move_cost(c) for c in node.children])
    
    def align2(self, trace: List[str], log_move_costs:Optional[List[int]], all_optimal=False, debug=False, timeout=None):
        """
        Main A* algorithm.
        Returns a list of goal states and the paths to reach them. state.path is the skip alignment you may look for.

        trace: List of activities
        log_move_costs: List of integers for the costs of log moves on the activities of trace. Requires len(log_move_costs) == len(trace)
        all_optimal: If True, returns all optimal skip alignments in normal form, otherwise one. Default: False
        debug: If True, prints the search stack. Default: False
        timeout: If not None, this is the maximal computation time in s

        Returns: 
            if all_optimal == True:
                (list of states for optimal skip alignments in normal form, computation time in ns)
            else:
                state for a optimal skip alignment in normal form
        """
        openlist:queue.PriorityQueue[State] = queue.PriorityQueue()
        closedset = []
        openlist.put(State.initial_state(self.tree, trace, self.mapper))

        tau_cost = self.max_tau_cost(self.tree)
        activity_cost = min(min(log_move_costs), self.min_model_move_cost(self.tree))
        assert tau_cost < activity_cost

        cost_bound = activity_cost-1

        optimal_states:List[State] = []

        time_start = time.process_time()
        time_start_ns = time.process_time_ns()

        while not openlist.empty():
            if timeout is not None and time.process_time() - time_start > timeout:
                return optimal_states, -1
            if debug:
                print()
                print("Open:  ", openlist.queue)
                print("Closed:", closedset)
            state = openlist.get()
            if debug:
                print("Inspecting state", state)
            if state.is_final():
                if debug:
                    print("State is final.")
                if all_optimal:
                    if len(optimal_states) == 0:
                        # first one is surely optimal
                        optimal_states.append(state)
                    elif state.acc_costs <= (optimal_states[0].acc_costs // activity_cost) * activity_cost + cost_bound:
                        # more expensive (probably) but still optimal
                        optimal_states.append(state)
                    else:
                        # not optimal anymore
                        return optimal_states, time.process_time_ns()-time_start_ns
                else:
                    return state
            else:
                other = self.find_in_set(state, closedset)
                if other is None or other.costs() >= state.costs(): # NOTE: >= is needed only if we want all optimal alignments ???
                    if debug:
                        print("Expanding state since other is", other)
                    if other is not None:
                        # remove to only have one in
                        closedset.remove(other)
                    closedset.append(state)
                    # do expansion
                    if len(optimal_states) > 0 and state.acc_costs > optimal_states[0].acc_costs:
                        # can never get optimal anymore
                        continue
                    successors = state.successors2(log_move_costs, time_start+timeout)
                    if debug:
                        print("Successors:", len(successors))
                    for s in successors:
                        openlist.put(s)
                        if debug:
                            print(" ", s)
                else:
                    if debug:
                        print("Already inspected (better) state, skip.")
        return optimal_states, time.process_time_ns() - time_start_ns
    
    def find_in_set(self, state:State, set:Set) -> State|None:
        for other in set:
            if state.matches_without_cost(other):
                return other
        return None
    


class Mapper(object):
    """
    Efficient mapper to access the nodes of a tree in a list.
    """

    def __init__(self, tree:ProcessTree) -> None:
        self.tree = tree
        self.order = self.traverse_nodes(self.tree)
        self.reverse_lookup = {t:i for i, t in enumerate(self.order)}
    
    def traverse_nodes(self, tree:ProcessTree):
        if isinstance(tree, LeafNode):
            return [tree]
        children = []
        for c in tree.children:
            children += self.traverse_nodes(c)
        return [tree] + children
    
    def node_to_index(self, tree:ProcessTree):
        return self.order.index(tree)
        #return self.reverse_lookup[tree]
    
    def index_to_node(self, index:int) -> ProcessTree:
        # assert 0 <= index and index < len(self.order)
        return self.order[index]
    
    def size(self):
        return len(self.order)
