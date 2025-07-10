import os
import sys
import pickle

from apis.agents.for_planner import AgentPlanner

with open('for_init.pkl', 'rb') as f:
    for_init = pickle.load(f)

with open('for_input.pkl', 'rb') as f:
    for_input = pickle.load(f)

p = AgentPlanner(**for_init)

final_path, heuristic_figs = p.plan_with_time_limit(for_input['map_bev'], for_input['start_xyt'], for_input['end_xyt'], time_limit=10, raise_exception=False, debug=True)