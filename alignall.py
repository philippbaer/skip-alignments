import pm4py
from processtree import *
from alignment import *
import pandas as pd
from tqdm import tqdm

from copy import copy

from pm4py.algo.conformance.alignments.petri_net import variants
from pm4py.objects.petri_net.utils import align_utils, check_soundness
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.util.xes_constants import DEFAULT_NAME_KEY, DEFAULT_TRACEID_KEY
from pm4py.objects.log.obj import Trace, Event
import time
from pm4py.util.lp import solver
from pm4py.util import exec_utils
from enum import Enum
import sys
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY, PARAMETER_CONSTANT_CASEID_KEY, CASE_CONCEPT_NAME
import importlib.util
from typing import Optional, Dict, Any, Union
from pm4py.objects.log.obj import EventLog, EventStream, Trace
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.util import typing, constants, pandas_utils
import pandas as pd
import heapq
import sys
import time
from copy import copy
from enum import Enum

import numpy as np

from pm4py.objects.log import obj as log_implementation
from pm4py.objects.petri_net.utils import align_utils as utils
from pm4py.objects.petri_net.utils.incidence_matrix import construct as inc_mat_construct
from pm4py.objects.petri_net.utils.synchronous_product import construct_cost_aware, construct
from pm4py.objects.petri_net.utils.petri_utils import construct_trace_net_cost_aware, decorate_places_preset_trans, \
    decorate_transitions_prepostset
from pm4py.util import exec_utils
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY
from pm4py.util.lp import solver as lp_solver
from pm4py.util.xes_constants import DEFAULT_NAME_KEY
from pm4py.util import variants_util
from typing import Optional, Dict, Any, Union
from pm4py.objects.log.obj import Trace
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.util import typing

"""
This file boxes all algorithms to compute alignments (PM4py) and skip alignments.
"""

############ PM4py like algorithm ############
# Modified to compute all optimal alignments #
##############################################
def apply_multiprocessing(log, petri_net, initial_marking, final_marking, id_loop_list, parameters=None, variant=pm4py.algo.conformance.alignments.petri_net.algorithm.DEFAULT_VARIANT, tree=None):
    if parameters is None:
        parameters = {}

    import multiprocessing

    variant = pm4py.algo.conformance.alignments.petri_net.algorithm.__variant_mapper(variant)

    num_cores = exec_utils.get_param_value(pm4py.algo.conformance.alignments.petri_net.algorithm.Parameters.CORES, parameters, multiprocessing.cpu_count() - 2)

    enable_best_worst_cost = exec_utils.get_param_value(pm4py.algo.conformance.alignments.petri_net.algorithm.Parameters.ENABLE_BEST_WORST_COST, parameters, True)

    variants_idxs, one_tr_per_var = pm4py.algo.conformance.alignments.petri_net.algorithm.__get_variants_structure(log, parameters)
    variant_strings = []
    for trace in one_tr_per_var:
        variant_strings.append([evt['concept:name'] for evt in trace._list])

    if enable_best_worst_cost:
        best_worst_cost = pm4py.algo.conformance.alignments.petri_net.algorithm.__get_best_worst_cost(petri_net, initial_marking, final_marking, variant, parameters)
        parameters[pm4py.algo.conformance.alignments.petri_net.algorithm.Parameters.BEST_WORST_COST_INTERNAL] = best_worst_cost

    all_alignments = {}

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = []
        for i, trace in enumerate(one_tr_per_var):
            futures.append(executor.submit(apply_trace, trace, petri_net, initial_marking, final_marking, id_loop_list, tree.get_cheapest_execution(0)[0]+len(variant_strings[i])*100000+0.1, parameters, str(variant)))
        progress = pm4py.algo.conformance.alignments.petri_net.algorithm.__get_progress_bar(len(one_tr_per_var), parameters)
        if progress is not None:
            alignments_ready = 0
            while alignments_ready != len(futures):
                current = 0
                for index, variant in enumerate(futures):
                    current = current + 1 if futures[index].done() else current
                if current > alignments_ready:
                    for i in range(0, current - alignments_ready):
                        progress.update()
                alignments_ready = current
        for index, variant in enumerate(futures):
            all_alignments[", ".join(variant_strings[index])] = futures[index].result()
        pm4py.algo.conformance.alignments.petri_net.algorithm.__close_progress_bar(progress)

    #alignments = __form_alignments(variants_idxs, all_alignments)

    return all_alignments

def apply_trace(trace, petri_net, initial_marking, final_marking, id_loop_list, cost_bound=10**8, parameters=None,
                variant=pm4py.algo.conformance.alignments.petri_net.algorithm.DEFAULT_VARIANT, all=True):
    time_start = time.process_time_ns()
    if parameters is None:
        parameters = copy({PARAMETER_CONSTANT_ACTIVITY_KEY: DEFAULT_NAME_KEY})

    variant = pm4py.algo.conformance.alignments.petri_net.algorithm.__variant_mapper(variant)
    parameters = copy(parameters)

    enable_best_worst_cost = exec_utils.get_param_value(pm4py.algo.conformance.alignments.petri_net.algorithm.Parameters.ENABLE_BEST_WORST_COST, parameters, True)

    ali = apply(trace, petri_net, initial_marking, final_marking, id_loop_list, cost_bound,
                                                 parameters=parameters, all=all)

    trace_cost_function = exec_utils.get_param_value(pm4py.algo.conformance.alignments.petri_net.algorithm.Parameters.PARAM_TRACE_COST_FUNCTION, parameters, [])

    # Instead of using the length of the trace, use the sum of the trace cost function
    time_end = time.process_time_ns()-time_start
    
    return time_end, ali

def apply_log(log, petri_net, initial_marking, final_marking, id_loop_list, cost_bound, parameters=None, variant=pm4py.algo.conformance.alignments.petri_net.algorithm.DEFAULT_VARIANT, all=True):
    if parameters is None:
        parameters = dict()

    if solver.DEFAULT_LP_SOLVER_VARIANT is not None:
        if not check_soundness.check_easy_soundness_net_in_fin_marking(petri_net, initial_marking, final_marking):
            raise Exception("trying to apply alignments on a Petri net that is not a easy sound net!!")

    enable_best_worst_cost = exec_utils.get_param_value(pm4py.algo.conformance.alignments.petri_net.algorithm.Parameters.ENABLE_BEST_WORST_COST, parameters, True)

    variant = pm4py.algo.conformance.alignments.petri_net.algorithm.__variant_mapper(variant)

    start_time = time.time()
    max_align_time = exec_utils.get_param_value(pm4py.algo.conformance.alignments.petri_net.algorithm.Parameters.PARAM_MAX_ALIGN_TIME, parameters,
                                                sys.maxsize)
    max_align_time_case = exec_utils.get_param_value(pm4py.algo.conformance.alignments.petri_net.algorithm.Parameters.PARAM_MAX_ALIGN_TIME_TRACE, parameters,
                                                     sys.maxsize)

    variants_idxs, one_tr_per_var = pm4py.algo.conformance.alignments.petri_net.algorithm.__get_variants_structure(log, parameters)
    progress = pm4py.algo.conformance.alignments.petri_net.algorithm.__get_progress_bar(len(one_tr_per_var), parameters)

    if enable_best_worst_cost:
        best_worst_cost = pm4py.algo.conformance.alignments.petri_net.algorithm.__get_best_worst_cost(petri_net, initial_marking, final_marking, variant, parameters)
        parameters[pm4py.algo.conformance.alignments.petri_net.algorithm.Parameters.BEST_WORST_COST_INTERNAL] = best_worst_cost

    all_alignments = []
    for trace in one_tr_per_var:
        this_max_align_time = min(max_align_time_case, (max_align_time - (time.time() - start_time)) * 0.5)
        parameters[pm4py.algo.conformance.alignments.petri_net.algorithm.Parameters.PARAM_MAX_ALIGN_TIME_TRACE] = this_max_align_time
        all_alignments.append(apply_trace(trace, petri_net, initial_marking, final_marking, id_loop_list, cost_bound, parameters=copy(parameters),
                                          variant=variant, all=all))
        if progress is not None:
            progress.update()

    #alignments = pm4py.algo.conformance.alignments.petri_net.algorithm.__form_alignments(variants_idxs, all_alignments)
    pm4py.algo.conformance.alignments.petri_net.algorithm.__close_progress_bar(progress)

    return all_alignments

def apply(trace: Trace, petri_net: PetriNet, initial_marking: Marking, final_marking: Marking, id_loop_list, cost_bound, parameters: Optional[Dict[Union[str, pm4py.algo.conformance.alignments.petri_net.variants.state_equation_a_star.Parameters], Any]] = None, all=True) -> typing.AlignmentResult:
    if parameters is None:
        parameters = {}

    activity_key = exec_utils.get_param_value(pm4py.algo.conformance.alignments.petri_net.variants.state_equation_a_star.Parameters.ACTIVITY_KEY, parameters, DEFAULT_NAME_KEY)
    trace_cost_function = exec_utils.get_param_value(pm4py.algo.conformance.alignments.petri_net.variants.state_equation_a_star.Parameters.PARAM_TRACE_COST_FUNCTION, parameters, None)
    model_cost_function = exec_utils.get_param_value(pm4py.algo.conformance.alignments.petri_net.variants.state_equation_a_star.Parameters.PARAM_MODEL_COST_FUNCTION, parameters, None)
    trace_net_constr_function = exec_utils.get_param_value(pm4py.algo.conformance.alignments.petri_net.variants.state_equation_a_star.Parameters.TRACE_NET_CONSTR_FUNCTION, parameters,
                                                           None)
    trace_net_cost_aware_constr_function = exec_utils.get_param_value(pm4py.algo.conformance.alignments.petri_net.variants.state_equation_a_star.Parameters.TRACE_NET_COST_AWARE_CONSTR_FUNCTION,
                                                                      parameters, construct_trace_net_cost_aware)
    
    if trace_cost_function is None:
        trace_cost_function = list(
            map(lambda e: utils.STD_MODEL_LOG_MOVE_COST, trace))
        parameters[pm4py.algo.conformance.alignments.petri_net.variants.state_equation_a_star.Parameters.PARAM_TRACE_COST_FUNCTION] = trace_cost_function

    if model_cost_function is None:
        # reset variables value
        model_cost_function = dict()
        sync_cost_function = dict()
        for t in petri_net.transitions:
            if t.label is not None:
                model_cost_function[t] = utils.STD_MODEL_LOG_MOVE_COST
                sync_cost_function[t] = utils.STD_SYNC_COST
            else:
                model_cost_function[t] = utils.STD_TAU_COST
        parameters[pm4py.algo.conformance.alignments.petri_net.variants.state_equation_a_star.Parameters.PARAM_MODEL_COST_FUNCTION] = model_cost_function
        parameters[pm4py.algo.conformance.alignments.petri_net.variants.state_equation_a_star.Parameters.PARAM_SYNC_COST_FUNCTION] = sync_cost_function

    if trace_net_constr_function is not None:
        print("Is old trace net function")
        # keep the possibility to pass TRACE_NET_CONSTR_FUNCTION in this old version
        trace_net, trace_im, trace_fm = trace_net_constr_function(trace, activity_key=activity_key)
    else:
        trace_net, trace_im, trace_fm, parameters[
            pm4py.algo.conformance.alignments.petri_net.variants.state_equation_a_star.Parameters.PARAM_TRACE_NET_COSTS] = trace_net_cost_aware_constr_function(trace,
                                                                                     trace_cost_function,
                                                                                     activity_key=activity_key)

    alignment = apply_trace_net(petri_net, initial_marking, final_marking, trace_net, trace_im, trace_fm, id_loop_list, cost_bound, parameters, all=all)

    return alignment

def apply_trace_net(petri_net, initial_marking, final_marking, trace_net, trace_im, trace_fm, id_loop_list, cost_bound, parameters=None, all=True):
    if parameters is None:
        parameters = {}

    ret_tuple_as_trans_desc = exec_utils.get_param_value(pm4py.algo.conformance.alignments.petri_net.variants.state_equation_a_star.Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE,
                                                         parameters, False)

    trace_cost_function = exec_utils.get_param_value(pm4py.algo.conformance.alignments.petri_net.variants.state_equation_a_star.Parameters.PARAM_TRACE_COST_FUNCTION, parameters, None)
    model_cost_function = exec_utils.get_param_value(pm4py.algo.conformance.alignments.petri_net.variants.state_equation_a_star.Parameters.PARAM_MODEL_COST_FUNCTION, parameters, None)
    sync_cost_function = exec_utils.get_param_value(pm4py.algo.conformance.alignments.petri_net.variants.state_equation_a_star.Parameters.PARAM_SYNC_COST_FUNCTION, parameters, None)
    trace_net_costs = exec_utils.get_param_value(pm4py.algo.conformance.alignments.petri_net.variants.state_equation_a_star.Parameters.PARAM_TRACE_NET_COSTS, parameters, None)

    if trace_cost_function is None or model_cost_function is None or sync_cost_function is None:
        sync_prod, sync_initial_marking, sync_final_marking = construct(trace_net, trace_im,
                                                                        trace_fm, petri_net,
                                                                        initial_marking,
                                                                        final_marking,
                                                                        utils.SKIP)
        cost_function = utils.construct_standard_cost_function(sync_prod, utils.SKIP)
    else:
        revised_sync = dict()
        for t_trace in trace_net.transitions:
            for t_model in petri_net.transitions:
                if t_trace.label == t_model.label:
                    revised_sync[(t_trace, t_model)] = sync_cost_function[t_model]

        sync_prod, sync_initial_marking, sync_final_marking, cost_function = construct_cost_aware(
            trace_net, trace_im, trace_fm, petri_net, initial_marking, final_marking, utils.SKIP,
            trace_net_costs, model_cost_function, revised_sync)

    max_align_time_trace = exec_utils.get_param_value(pm4py.algo.conformance.alignments.petri_net.variants.state_equation_a_star.Parameters.PARAM_MAX_ALIGN_TIME_TRACE, parameters,
                                                      sys.maxsize)

    alignment = apply_sync_prod(sync_prod, sync_initial_marking, sync_final_marking, cost_function,
                           utils.SKIP, id_loop_list, cost_bound, ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                           max_align_time_trace=max_align_time_trace, all=all)

    return_sync_cost = exec_utils.get_param_value(pm4py.algo.conformance.alignments.petri_net.variants.state_equation_a_star.Parameters.RETURN_SYNC_COST_FUNCTION, parameters, False)
    if return_sync_cost:
        # needed for the decomposed alignments (switching them from state_equation_less_memory)
        return alignment, cost_function

    return alignment

def apply_sync_prod(sync_prod, initial_marking, final_marking, cost_function, skip, id_loop_list, cost_bound, ret_tuple_as_trans_desc=False,
                    max_align_time_trace=sys.maxsize, all=True):
    return __search(sync_prod, initial_marking, final_marking, cost_function, skip, id_loop_list, cost_bound,
                    ret_tuple_as_trans_desc=ret_tuple_as_trans_desc, max_align_time_trace=max_align_time_trace, all=all)

def is_closed(state:utils.SearchTuple, closed:List, ret_tuple_as_trans_desc:bool):
    agn = __reconstruct_alignment(state, 0, 0, 0,
                                                     ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                                                     lp_solved=0)
    search_tuple = (agn["alignment"], state.m)
    return search_tuple in closed

def does_allow_tau_path(process_tree:pm4py.objects.process_tree.obj.ProcessTree):
    cond1 = False
    cond2 = False
    tree = ProcessTree.from_pm4py(process_tree, 100000, 0, 0)
    assert isinstance(tree, Loop)
    if tree.children[0].get_cheapest_execution(0)[1]:
        # allows for a tau path
        cond1 = True
    for c in tree.children[1:]:
        if c.get_cheapest_execution(0)[1]:
            # a redo with tau path does exist
            cond2 = True
    return cond1 and cond2

def insert_cycle_checks(process_tree:pm4py.objects.process_tree.obj.ProcessTree, id="1"):
    """
    Method needed to ensure that all computed alignments are free from cycles. The process model is modified for loops that allow tau-loops.
    """
    if process_tree.operator is None:
        return [], process_tree
    if process_tree.operator != pm4py.objects.process_tree.obj.Operator.LOOP or not does_allow_tau_path(process_tree):
        res = []
        for i, c in enumerate(process_tree.children):
            res += insert_cycle_checks(c, id+str(i))[0]
        return res, process_tree
    parent = process_tree.parent
    entry = pm4py.objects.process_tree.obj.ProcessTree(None, None, None, "TAU_entry_"+id)
    exit = pm4py.objects.process_tree.obj.ProcessTree(None, None, None, "TAU_exit_"+id)
    sequence = pm4py.objects.process_tree.obj.ProcessTree(pm4py.objects.process_tree.obj.Operator.SEQUENCE, parent, [entry, process_tree, exit], None)
    entry.parent = sequence
    exit.parent = sequence

    if parent is not None:
        loop_idx = parent.children.index(process_tree)
        parent.children[loop_idx] = sequence
    process_tree.parent = sequence
    res = []
    for i, c in enumerate(process_tree.children):
        res += insert_cycle_checks(c, id+str(i))[0]
    return [(id, process_tree)] + res, process_tree if parent is not None else sequence

def is_cycling(state:utils.SearchTuple, ret_tuple_as_trans_desc:bool, id_tree_list:List):
    agn = __reconstruct_alignment(state, 0, 0, 0,
                                                     ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                                                     lp_solved=0)
    agn['alignment'] = [a.label for a in agn['alignment'] if a.label[1] is not None]
    for id, loop in id_tree_list:
        if is_cycling_exec(agn['alignment'], loop, id):
            return True
    return False

def is_cycling_exec(agn:List, loop:pm4py.objects.process_tree.obj.ProcessTree, id:str):
    do_leafs = get_leafs(loop.children[0])
    redo_leafs = []
    for c in loop.children[1:]:
        redo_leafs += get_leafs(c)
    
    loop_executions = []
    for i in range(len(agn)):
        if agn[i][1] == "TAU_entry_"+id:
            need_to_add = True
            for j in range(i,len(agn)):
                if agn[j][1] == "TAU_exit_"+id:
                    need_to_add = False
                    loop_executions.append(agn[i+1:j+1]) # we add the TAU_exit to know that the last execution child is valid
                    break
            if need_to_add and len(agn[i+1:]) > 0:
                # still running execution of the loop
                loop_executions.append(agn[i+1:])
    
    for execution in loop_executions:
        executions_tau = []
        current = None
        current_tau = True
        for l,m in execution:
            if len(executions_tau) >= 2 and executions_tau[-1] and executions_tau[-2]:
                return True
            if m in do_leafs:
                if current is None:
                    current = "do"
                    current_tau = m.startswith("TAU_")
                    executions_tau.append(False)
                elif current == "redo":
                    # finish up the redo
                    executions_tau[-1] = current_tau
                    current = "do"
                    current_tau = m.startswith("TAU_")
                    executions_tau.append(False)
                else:
                    # continuation of a do
                    if not m.startswith("TAU_"):
                        current_tau = False
            elif m in redo_leafs:
                if current == "do":
                    # finish up the do
                    executions_tau[-1] = current_tau
                    current = "redo"
                    current_tau = m.startswith("TAU_")
                    executions_tau.append(False)
                else:
                    # continuation of a redo
                    if not m.startswith("TAU_"):
                        current_tau = False

        if len(executions_tau) > 0:
            executions_tau[-1] = current_tau
        if execution[-1][1] == "TAU_exit_"+id:
            # the last added boolean in executions_tau is valid
            pass
        else:
            # we don't know if the last child of the execution will later on contain an activity
            executions_tau = executions_tau[:-1]
        if len(executions_tau) >= 2 and executions_tau[-1] and executions_tau[-2]:
            return True
    return False

def get_leafs(process_tree:pm4py.objects.process_tree.obj.ProcessTree):
    if process_tree.operator == None:
        # we use our translated tree to pm4py, i.e., taus get activities
        assert process_tree.label is not None
        return [process_tree.label]
    res = []
    for c in process_tree.children:
        res += get_leafs(c)
    return res

def __search(sync_net, ini, fin, cost_function, skip, id_loop_list, cost_bound, ret_tuple_as_trans_desc=False,
             max_align_time_trace=sys.maxsize, all=True):
    """
    Performs A* search for (all) optimal alignments.
    """
    start_time = time.process_time()
    time_for_first_alignment = -1
    first_time = time.process_time_ns()

    opt_cost = 10**10
    opt_agns = []

    decorate_transitions_prepostset(sync_net)
    decorate_places_preset_trans(sync_net)

    incidence_matrix = inc_mat_construct(sync_net)
    ini_vec, fin_vec, cost_vec = utils.__vectorize_initial_final_cost(incidence_matrix, ini, fin, cost_function)

    closed = list()

    a_matrix = np.asmatrix(incidence_matrix.a_matrix).astype(np.float64)
    g_matrix = -np.eye(len(sync_net.transitions))
    h_cvx = np.matrix(np.zeros(len(sync_net.transitions))).transpose()
    cost_vec = [x * 1.0 for x in cost_vec]

    use_cvxopt = False
    if lp_solver.DEFAULT_LP_SOLVER_VARIANT == lp_solver.CVXOPT_SOLVER_CUSTOM_ALIGN or lp_solver.DEFAULT_LP_SOLVER_VARIANT == lp_solver.CVXOPT_SOLVER_CUSTOM_ALIGN_ILP:
        use_cvxopt = True

    if use_cvxopt:
        # not available in the latest version of PM4Py
        from cvxopt import matrix

        a_matrix = matrix(a_matrix)
        g_matrix = matrix(g_matrix)
        h_cvx = matrix(h_cvx)
        cost_vec = matrix(cost_vec)

    h, x = utils.__compute_exact_heuristic_new_version(sync_net, a_matrix, h_cvx, g_matrix, cost_vec, incidence_matrix,
                                                       ini,
                                                       fin_vec, lp_solver.DEFAULT_LP_SOLVER_VARIANT,
                                                       use_cvxopt=use_cvxopt)
    ini_state = utils.SearchTuple(0 + h, 0, h, ini, None, None, x, True)
    open_set = [ini_state]
    heapq.heapify(open_set)
    visited = 0
    queued = 0
    traversed = 0
    lp_solved = 1

    trans_empty_preset = set(t for t in sync_net.transitions if len(t.in_arcs) == 0)

    while not len(open_set) == 0:
        if (time.process_time() - start_time) > max_align_time_trace:
            return reset_results(opt_agns), reset_time(opt_agns, -1), time_for_first_alignment
        curr = heapq.heappop(open_set)

        current_marking = curr.m

        while not curr.trust:
            if (time.process_time() - start_time) > max_align_time_trace:
                return reset_results(opt_agns), reset_time(opt_agns, -1), time_for_first_alignment

            already_closed = is_closed(curr, closed, ret_tuple_as_trans_desc)
            if already_closed:
                curr = heapq.heappop(open_set)
                current_marking = curr.m
                continue

            h, x = utils.__compute_exact_heuristic_new_version(sync_net, a_matrix, h_cvx, g_matrix, cost_vec,
                                                               incidence_matrix, curr.m,
                                                               fin_vec, lp_solver.DEFAULT_LP_SOLVER_VARIANT,
                                                               use_cvxopt=use_cvxopt)
            lp_solved += 1

            # 11/10/19: shall not a state for which we compute the exact heuristics be
            # by nature a trusted solution?
            tp = utils.SearchTuple(curr.g + h, curr.g, h, curr.m, curr.p, curr.t, x, True)
            # 11/10/2019 (optimization ZA) heappushpop is slightly more efficient than pushing
            # and popping separately
            curr = heapq.heappushpop(open_set, tp)
            current_marking = curr.m

        #print("Current state after update:", curr)
        # max allowed heuristics value (27/10/2019, due to the numerical instability of some of our solvers)
        if curr.h > lp_solver.MAX_ALLOWED_HEURISTICS:
            continue

        # 12/10/2019: do it again, since the marking could be changed
        already_closed = is_closed(curr, closed, ret_tuple_as_trans_desc)
        if already_closed:
            continue

        # 12/10/2019: the current marking can be equal to the final marking only if the heuristics
        # (underestimation of the remaining cost) is 0. Low-hanging fruits
        if curr.h < 0.01:
            if current_marking == fin:
                agn = __reconstruct_alignment(curr, visited, queued, traversed,
                                                     ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                                                     lp_solved=lp_solved)
                #print(str(len(opt_agns)), "State is final:", agn['alignment'])
                if agn['cost'] <= opt_cost:
                    opt_cost = agn['cost']
                    opt_agns.append(agn)

                    if len(opt_agns) == 1:
                        time_for_first_alignment = time.process_time_ns()-first_time
                    if not all:
                        # no reset as the first alignment is fine even with cycling
                        return opt_agns, 0, time_for_first_alignment
                    continue
                else:
                    return reset_results(opt_agns), reset_time(opt_agns, 0), time_for_first_alignment
                
        closed.append((__reconstruct_alignment(curr, visited, queued, traversed,
                                                     ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                                                     lp_solved=lp_solved)['alignment'], curr.m))
        visited += 1

        enabled_trans = copy(trans_empty_preset)
        for p in current_marking:
            for t in p.ass_trans:
                if t.sub_marking <= current_marking:
                    enabled_trans.add(t)
        trans_to_visit_with_cost = [(t, cost_function[t]) for t in enabled_trans if not (
                t is not None and utils.__is_log_move(t, skip) and utils.__is_model_move(t, skip))]

        for t, cost in trans_to_visit_with_cost:
            traversed += 1
            new_marking = utils.add_markings(current_marking, t.add_marking)

            g = curr.g + cost

            queued += 1
            h, x = utils.__derive_heuristic(incidence_matrix, cost_vec, curr.x, t, curr.h)
            trustable = utils.__trust_solution(x)
            new_f = g + h

            tp = utils.SearchTuple(new_f, g, h, new_marking, curr, t, x, trustable)
            if not g > opt_cost and not new_f > cost_bound and not is_closed(tp, closed, ret_tuple_as_trans_desc) and not is_cycling(tp, ret_tuple_as_trans_desc, id_loop_list):
                heapq.heappush(open_set, tp)
    return reset_results(opt_agns), reset_time(opt_agns, 0), time_for_first_alignment

def reset_results(agns):
    # annulates the result if there are infinitely many optimal alignments
    for a in agns:
        for t in a['alignment']:
            if t.label[1] is not None and t.label[1].startswith("TAU_entry"):
                return []
    return agns

def reset_time(agns, timing):
    # annulates the timing state to timeout if there are infinitely many optimal alignments
    for a in agns:
        for t in a['alignment']:
            if t.label[1] is not None and t.label[1].startswith("TAU_entry"):
                return -1
    return timing

def __reconstruct_alignment(state, visited, queued, traversed, ret_tuple_as_trans_desc=False, lp_solved=0):
    alignment = list()
    if state.p is not None and state.t is not None:
        parent = state.p
        if ret_tuple_as_trans_desc:
            alignment = [state.t]
            while parent.p is not None:
                alignment = [parent.t] + alignment
                parent = parent.p
        else:
            alignment = [state.t]
            while parent.p is not None:
                alignment = [parent.t] + alignment
                parent = parent.p
    return {'alignment': alignment, 'cost': state.g, 'visited_states': visited, 'queued_states': queued,
            'traversed_arcs': traversed, 'lp_solved': lp_solved}

def align_pn_all(var:List[str], net, init, final, id_loop_list, cost_bound=10**8, timeout=100):
    """
    Computes all optimal alignments free from cycles. It does not use multiprocessing.

    var: List of activities that represents a trace
    net: Petri net structure
    init: Initial marking of the Petri net
    final: Final marking of the Petri net
    id_loop_list: List of ids of loop nodes that potentially cycle
    cost_bound: Upper limit for the costs an optimal alignment can have
    timeout: Maximal computation time in s

    Returns: Dict that maps variant string to tuples (total computation time in ns, (list of optimal alignments, -1 for timeout otherwise 0, computation time for the first optimal alignment in ns))
    """
    def net_model_move(t):
        if t.label is None:
            return 0
        if t.label.startswith("TAU"):
            return 0
        return 100000

    param = {
        'trace_cost_function': [100000]*len(var),
        'model_cost_function': {t:net_model_move(t) for t in net.transitions},
        'sync_cost_function':  {t:0 for t in net.transitions},
        pm4py.algo.conformance.alignments.petri_net.variants.state_equation_a_star.Parameters.PARAM_MAX_ALIGN_TIME_TRACE: timeout
    }
    var_df = pd.DataFrame({'case:concept:name': '1', 'concept:name': var, 'time:timestamp': [pd.Timestamp(year=1000+i, month=1, day=1) for i, _ in enumerate(var)]})
    #var_df = log[log['case:concept:name'] == 'A100']
    #agn = apply_trace(var_df, net, init, final, [], parameters=param, variant=pm4py.algo.conformance.alignments.petri_net.algorithm.Variants.VERSION_STATE_EQUATION_A_STAR)
    return apply_log(var_df, net, init, final, id_loop_list, cost_bound, parameters=param, variant=pm4py.algo.conformance.alignments.petri_net.algorithm.Variants.VERSION_STATE_EQUATION_A_STAR)

def align_pn_all_multi(log, net, init, final, id_loop_list, tree=None, timeout=100):
    """
    Computes all optimal alignments free from cycles. It uses multiprocessing.

    log: PM4py event log
    net: Petri net structure
    init: Initial marking of the Petri net
    final: Final marking of the Petri net
    id_loop_list: List of ids of loop nodes that potentially cycle
    tree: Process tree
    timeout: Maximal computation time in s

    Returns: Dict that maps variant string to tuples (total computation time in ns, (list of optimal alignments, -1 for timeout otherwise 0, computation time for the first optimal alignment in ns))
    """
    def net_model_move(t):
        if t.label is None:
            return 0
        if t.label.startswith("TAU"):
            return 0
        return 100000

    param = {
        'trace_cost_function': [100000]*1000,
        'model_cost_function': {t:net_model_move(t) for t in net.transitions},
        'sync_cost_function':  {t:0 for t in net.transitions},
        pm4py.algo.conformance.alignments.petri_net.variants.state_equation_a_star.Parameters.PARAM_MAX_ALIGN_TIME_TRACE: timeout,
        'cores': 14
    }
    return apply_multiprocessing(log, net, init, final, id_loop_list, parameters=param, variant=pm4py.algo.conformance.alignments.petri_net.algorithm.Variants.VERSION_STATE_EQUATION_A_STAR, tree=tree)

def align_pn_all_for_one(var:List[str], net, init, final, id_loop_list, cost_bound=10**8, timeout=100):
    """
    Computes all optimal alignments free from cycles. It does not use multiprocessing.

    var: List of activities that represents a trace
    net: Petri net structure
    init: Initial marking of the Petri net
    final: Final marking of the Petri net
    id_loop_list: List of ids of loop nodes that potentially cycle
    cost_bound: Upper limit for the costs an optimal alignment can have
    timeout: Maximal computation time in s

    Returns: Computation time in ns
    """
    def net_model_move(t):
        if t.label is None:
            return 0
        if t.label.startswith("TAU"):
            return 0
        return 100000

    param = {
        'trace_cost_function': [100000]*len(var),
        'model_cost_function': {t:net_model_move(t) for t in net.transitions},
        'sync_cost_function':  {t:0 for t in net.transitions},
        pm4py.algo.conformance.alignments.petri_net.variants.state_equation_a_star.Parameters.PARAM_MAX_ALIGN_TIME_TRACE: timeout
    }
    var_df = pd.DataFrame({'case:concept:name': '1', 'concept:name': var, 'time:timestamp': [pd.Timestamp(year=1000+i, month=1, day=1) for i, _ in enumerate(var)]})
    time_start = time.process_time_ns()
    for _ in range(200):
        apply_log(var_df, net, init, final, id_loop_list, cost_bound, parameters=param, variant=pm4py.algo.conformance.alignments.petri_net.algorithm.Variants.VERSION_STATE_EQUATION_A_STAR)
    return (time.process_time_ns()-time_start)/200

def align_pn_one_for_one(var:List[str], net, init, final, id_loop_list, cost_bound=10**8, timeout=100, cnt=2000):
    """
    Computes one optimal alignment free from cycles. It does not use multiprocessing.

    var: List of activities that represents a trace
    net: Petri net structure
    init: Initial marking of the Petri net
    final: Final marking of the Petri net
    id_loop_list: List of ids of loop nodes that potentially cycle
    cost_bound: Upper limit for the costs an optimal alignment can have
    timeout: Maximal computation time in s
    cnt: Number of iterations to perform the computation of the alignment for precise timing

    Returns: Computation time in ns
    """
    def net_model_move(t):
        if t.label is None:
            return 0
        if t.label.startswith("TAU"):
            return 0
        return 100000

    param = {
        'trace_cost_function': [100000]*len(var),
        'model_cost_function': {t:net_model_move(t) for t in net.transitions},
        'sync_cost_function':  {t:0 for t in net.transitions},
        pm4py.algo.conformance.alignments.petri_net.variants.state_equation_a_star.Parameters.PARAM_MAX_ALIGN_TIME_TRACE: timeout
    }
    var_df = pd.DataFrame({'case:concept:name': '1', 'concept:name': var, 'time:timestamp': [pd.Timestamp(year=1000+i, month=1, day=1) for i, _ in enumerate(var)]})
    time_start = time.process_time_ns()
    for _ in range(cnt):
        apply_log(var_df, net, init, final, id_loop_list, cost_bound, parameters=param, variant=pm4py.algo.conformance.alignments.petri_net.algorithm.Variants.VERSION_STATE_EQUATION_A_STAR, all=False)
    return (time.process_time_ns()-time_start)/cnt

def align_sk_all_for_one(tree:ProcessTree, var:List[str], timeout=100, cnt=2000):
    """
    Computes all optimal skip alignments in normal form. It does not use multiprocessing.

    tree: Process tree to align to
    var: List of activities that represents a trace
    timeout: Maximal computation time in s
    cnt: Number of iterations to perform the computation of the alignment for precise timing

    Returns: Computation time in ns
    """
    Aligner.set_level_incentive(0)
    timer = time.process_time_ns()
    aligner = Aligner(tree)
    for _ in range(cnt):
        aligner.align2(list(var), [100000]*len(var), True, timeout=timeout)
    return (time.process_time_ns()-timer)/cnt 

def align_sk_all(variant_strings:List[List[str]], tree:ProcessTree, timeout=100):
    """
    Computes all optimal skip alignments in normal form. It does use multiprocessing.

    variant_strings: List of lists of activities that represent a trace
    tree: Process tree to align to
    timeout: Maximal computation time in s

    Returns: Futures of the tuple (list of states whose paths represent optimal skip alignments in normal form, computation time in ns or -1 on timeout)
    """
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=14) as executor:
        futures = []
        for var in variant_strings:
            futures.append(executor.submit(Aligner(tree).align2, list(var), [100000]*len(var), True, timeout=timeout))
        progress = tqdm(total=len(futures))
        if progress is not None:
            alignments_ready = 0
            while alignments_ready != len(futures):
                current = 0
                for index, variant in enumerate(futures):
                    current = current + 1 if futures[index].done() else current
                if current > alignments_ready:
                    for i in range(0, current - alignments_ready):
                        progress.update()
                alignments_ready = current
        return futures