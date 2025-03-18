import abc
from typing import List, Optional
import pm4py
import itertools
from functools import reduce
from itertools import combinations

class ProcessTree(object):
    """
    This class represents a process model in the form of an process tree, i.e., an SBWF-net.
    It can be a leaf node (Activity, Tau) or an inner node (Sequence, Xor, And, Loop).
    Pseudostructures are provided for skips (Sskip, TauPath).
    """
    __metaclass__ = abc.ABCMeta
    parent:Optional["ProcessTree"] = None
    children:List["ProcessTree"] = []

    id:str = ""

    def __init__(self, parent:"ProcessTree"=None, children:List["ProcessTree"]=[]) -> None:
        self.parent = parent
        self.children = children
        self.leafs = None
        self.leaf_labels = None
    
    def set_parent(self, parent:"ProcessTree"):
        self.parent = parent

    @staticmethod
    def from_pm4py(process_tree:pm4py.objects.process_tree.obj.ProcessTree, model_move_activity_cost:int, model_move_tau_cost:int, sync_move_cost:int, id:str="0"):
        """
        Creates a process tree from a pm4py ProcessTree object.
        """
        if process_tree.operator is None and (process_tree.label is None or process_tree.label.startswith("TAU_")):
            # tau
            node = Tau(None, "TAU", model_move_tau_cost)
            node.id = "TAU_"+id
            node.sync_move_cost = sync_move_cost
            return node
        if process_tree.operator is None and process_tree.label is not None:
            # activity
            node = Activity(None, process_tree.label, model_move_activity_cost)
            node.id = "ACTIVITY_"+id
            node.sync_move_cost = sync_move_cost
            return node
        # operator
        children = [ProcessTree.from_pm4py(c, model_move_activity_cost, model_move_tau_cost, sync_move_cost, id + str(i)) for i, c in enumerate(process_tree.children)]
        if process_tree.operator == pm4py.objects.process_tree.obj.Operator.SEQUENCE:
            node = Sequence(None, children)
        elif process_tree.operator == pm4py.objects.process_tree.obj.Operator.XOR:
            node = Xor(None, children)
        elif process_tree.operator == pm4py.objects.process_tree.obj.Operator.PARALLEL:
            node = And(None, children)
        elif process_tree.operator == pm4py.objects.process_tree.obj.Operator.LOOP:
            node = Loop(None, children)
        else:
            raise TypeError("Provided process tree does contain non-standard nodes")
        node.id = id
        for i in range(len(children)):
            children[i].set_parent(node)
        return node
    
    def to_pm4py(self):
        """
        Translates the current process tree into a representation for pm4py.
        """
        if isinstance(self, Tau):
            # tau
            return pm4py.objects.process_tree.obj.ProcessTree(None, None, None, "TAU_" + self.name)
        if isinstance(self, Activity):
            # activity
            return pm4py.objects.process_tree.obj.ProcessTree(None, None, None, self.name)
        assert not isinstance(self, LeafNode)
        children = [c.to_pm4py() for c in self.children]
        if isinstance(self, Sequence):
            node = pm4py.objects.process_tree.obj.ProcessTree(pm4py.objects.process_tree.obj.Operator.SEQUENCE, None, children, None)
        elif isinstance(self, Xor):
            node = pm4py.objects.process_tree.obj.ProcessTree(pm4py.objects.process_tree.obj.Operator.XOR, None, children, None)
        elif isinstance(self, And):
            node = pm4py.objects.process_tree.obj.ProcessTree(pm4py.objects.process_tree.obj.Operator.PARALLEL, None, children, None)
        elif isinstance(self, Loop):
            node = pm4py.objects.process_tree.obj.ProcessTree(pm4py.objects.process_tree.obj.Operator.LOOP, None, children, None)
        else:
            raise TypeError("Process tree does contain non-standard node", self)
        for c in children:
            c.parent = node
        return node

    def get_leaf_labels(self):
        """
        Returns a list of leaf node activity labels.
        """
        if self.leaf_labels is not None:
            return self.leaf_labels
        if isinstance(self, LeafNode):
            if isinstance(self, Activity):
                self.leaf_labels = [self.name]
                return [self.name]
            self.leaf_labels = []
            return []
        self.leaf_labels = []
        for c in self.children:
            self.leaf_labels += c.get_leaf_labels()
        return self.leaf_labels
    
    def get_leafs(self):
        """
        Returns a list of leaf nodes of the process tree.
        """
        if self.leafs is not None:
            return self.leafs
        if isinstance(self, LeafNode):
            self.leafs = [self]
            return self.leafs
        self.leafs = []
        for c in self.children:
            self.leafs += c.get_leafs()
        return self.leafs
    
    def contains_tree(self, tree:"ProcessTree"):
        """
        Returns true if the provided process tree is a subtree of this tree.
        """
        if self == tree:
            return True
        for c in self.children:
            if c.contains_tree(tree):
                return True
        return False
    
    @staticmethod
    def get_lca(child1:"ProcessTree", child2:"ProcessTree") -> "ProcessTree":
        """
        Returns the least common ancestor node in the current tree for the two provided trees child1 and child2.
        """        
        if child1 is None or child2 is None:
            return None
        if child1 == child2:
            return child1
        parents1 = [child1]
        tmp = child1
        while tmp.parent is not None:
            tmp = tmp.parent
            parents1.append(tmp)
        tmp = child2
        while tmp is not None:
            if tmp in parents1:
                return tmp
            tmp = tmp.parent
        return None

    @staticmethod
    def get_generalized_log_lcas(child1:"ProcessTree", child2:"ProcessTree") -> List["ProcessTree"]:
        """
        Returns the least common ancestor of child1 and child2 in the current tree and all its previous And nodes.
        """
        lcas = [ProcessTree.get_lca(child1, child2)]
        current = lcas[0].parent
        while current is not None:
            if isinstance(current, And):
                # we generalize to the highest And if needed
                lcas.append(current)
            current = current.parent
        lcas.reverse() # [highest lca, ..., lowest lca]
        return lcas
    
    def get_max_depth(self):
        """
        Returns the maximal depth of the current tree.
        """        
        if isinstance(self, Skip) or isinstance(self, TauPath):
            return self.node.get_max_depth()
        if isinstance(self, LeafNode):
            return 0
        else:
            max_depth = 0
            for c in self.children:
                max_depth = max(max_depth, c.get_max_depth())
            return max_depth+1
    
    def get_distance_to_root(self):
        """
        Returns the level of the current subtree.
        """
        if self.parent is None:
            return 0
        return self.parent.get_distance_to_root() + 1
    
    def _get_cheapest_execution(self, level_incentive:int):
        # returns (cost, only Taus?)
        if isinstance(self, Tau):
            #return (self.skip_cost, True)
            return (0, True) # TAUs produce no costs
        if isinstance(self, LeafNode):
            return (self.skip_cost, False)
        if isinstance(self, Sequence) or isinstance(self, And):
            children = [child._get_cheapest_execution(level_incentive) for child in self.children]
            return (sum(c[0] for c in children), min(c[1] for c in children))
        if isinstance(self, Xor):
            children = [child._get_cheapest_execution(level_incentive) for child in self.children]
            cheapest_index = -1
            for i in range(len(children)):
                if cheapest_index == -1 or children[cheapest_index][0] > children[i][0]:
                    cheapest_index = i
            return children[cheapest_index]
        if isinstance(self, Loop):
            return self.children[0]._get_cheapest_execution(level_incentive)
        raise ValueError("Cannot match type of node.", self.__str__(), type(self))
        return -100, False

    def get_cheapest_execution(self, level_incentive:int):
        """
        Returns a tuple (cost, taus) with the cost of the cheapest execution of the current tree and a Boolean taus = True iff the cheapest execution consists only of Tau nodes.
        """
        cheapest = self._get_cheapest_execution(level_incentive)
        if cheapest[1]:
            # only taus on the path, increase cost artificially
            return (cheapest[0] + self.get_max_depth() * level_incentive, True)
        else:
            # not only taus on the path, decrease cost artificially
            return (cheapest[0] - self.get_max_depth() * level_incentive, False)

    def __str__(self) -> str:
        "Abtract Tree"

    def __repr__(self) -> str:
        return self.__str__()
    

class Sequence(ProcessTree):
    """
    Sequence node →(N_1,...,N_k)
    """
    def __init__(self, parent:ProcessTree, children:List[ProcessTree]) -> None:
        super().__init__(parent, children)
    
    def __str__(self) -> str:
        child_string = "\n".join([c.__str__() for c in self.children])
        return (" " * self.get_distance_to_root()*2) + "→\n" + child_string
               

class Xor(ProcessTree):
    """
    Choice node ×(N_1,...,N_k)
    """
    def __init__(self, parent:ProcessTree, children:List[ProcessTree]) -> None:
        super().__init__(parent, children)
    
    def __str__(self) -> str:
        child_string = "\n".join([c.__str__() for c in self.children])
        return (" " * self.get_distance_to_root()*2) + "×\n" + child_string

class And(ProcessTree):
    """
    Parallel node ∧(N_1,...,N_k)
    """
    def __init__(self, parent:ProcessTree, children:List[ProcessTree]) -> None:
        super().__init__(parent, children)
    
    def __str__(self) -> str:
        child_string = "\n".join([c.__str__() for c in self.children])
        return (" " * self.get_distance_to_root()*2) + "∧\n" + child_string

class Loop(ProcessTree):
    """
    Loop node ↺(N_1,...,N_k)
    """
    def __init__(self, parent:ProcessTree, children:List[ProcessTree]) -> None:
        super().__init__(parent, children)
    
    def __str__(self) -> str:
        child_string = "\n".join([c.__str__() for c in self.children])
        return (" " * self.get_distance_to_root()*2) + "↺\n" + child_string

class LeafNode(ProcessTree):
    """
    Abstract representation of a leaf node
    """
    __metaclass__ = abc.ABCMeta
    name:str
    sync_move_cost:int = 0
    skip_cost:int = 0 # skipping the subtree or doing the model move

    def __init__(self, parent: ProcessTree, name:str, model_move_cost:int) -> None:
        super().__init__(parent, [])
        self.name = name
        self.skip_cost = model_move_cost

class Activity(LeafNode):
    """
    Activity node, i.e., a labelled non-tau transition in an SBWF-net
    """
    def __init__(self, parent:ProcessTree, name:str, model_move_cost:int) -> None:
        super().__init__(parent, name, model_move_cost)
    
    def __str__(self) -> str:
        return (" " * self.get_distance_to_root()*2) + "Act( " + self.name + " )"

class Tau(LeafNode):
    """
    Tau node, i.e., a labelled tau transition in an SBWF-net
    """
    def __init__(self, parent:ProcessTree, name:str, model_move_cost:int) -> None:
        super().__init__(parent, name, model_move_cost)
    
    def __str__(self) -> str:
        return (" " * self.get_distance_to_root()*2) + "Tau( " + self.name + " )"

class ArtificialTau(LeafNode):
    def __init__(self, parent:ProcessTree, name:str, model_move_cost:int) -> None:
        super().__init__(parent, name, model_move_cost)
    
    def __str__(self) -> str:
        return (" " * self.get_distance_to_root()*2) + "ArtTau( " + self.name + " )"

class Skip(ProcessTree):
    """
    Wrapper for a skip move with cost > 0
    """
    def __init__(self, node:ProcessTree, skip_cost:int) -> None:
        super().__init__(node.parent, [])
        self.node = node
        self.skip_cost = skip_cost
    
    def __str__(self) -> str:
        return "Skip( " + str(type(self.node)) + " )"
    
class TauPath(ProcessTree):
    """
    Wrapper for a skip move with cost = 0
    """
    def __init__(self, node:ProcessTree) -> None:
        super().__init__(node.parent, [])
        self.node = node
    
    def __str__(self) -> str:
        return "TauPath( " + str(type(self.node)) + " )"

    def unfold(self):
        if isinstance(self.node, Tau):
            return [[self.node]]
        elif isinstance(self.node, Sequence) or isinstance(self.node, And): # we fix the order of And executions
            paths = []
            for c in self.node.children:
                paths.append(TauPath(c).unfold())
            res = []
            for path in list(itertools.product(*paths)):
                res.append([t for elem in path for t in elem])
            return res
        elif isinstance(self.node, Xor):
            # we don't know yet which path contains a Tau
            paths = []
            for c in self.node.children:
                if c.get_cheapest_execution(0)[1]:
                    paths += TauPath(c).unfold()
            return paths
        elif isinstance(self.node, Loop):
            return TauPath(self.node.children[0]).unfold()
        assert ValueError("Unexpected type of node to unfold")

    def all_order_preserving_shuffles(self, *paths):
        inter = []
        for comb in itertools.combinations(range(len(paths[0])+len(paths[1])), len(paths[0])):
            inter.append(self.assign(paths[1], list(zip(comb, paths[0]))))
        if len(paths) == 2:
            return inter
        res = []
        for new_path in inter:
            res += self.all_order_preserving_shuffles(new_path, *paths[2:])
        return res

    def assign(self, fillup_list, insert_list):
        l = []
        for i in range(len(insert_list)+len(fillup_list)):
            if len(insert_list) > 0 and insert_list[0][0] == i:
                l.append(insert_list[0][1])
                insert_list = insert_list[1:]
            else:
                l.append(fillup_list[0])
                fillup_list = fillup_list[1:]
        return l
