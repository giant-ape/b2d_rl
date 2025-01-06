#!/bin/bash
BASE_PORT=3000
BASE_TM_PORT=5000
IS_BENCH2DRIVE=True
BASE_ROUTES=leaderboard/data/bench2drive9
TEAM_AGENT=leaderboard/team_code/vad_b2d_agent.py
#TEAM_CONFIG=your_team_agent_ckpt.pth   # for TCP and ADMLP
TEAM_CONFIG=/home/whut613/zjy/Bench2Drive/Bench2DriveZoo/adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py+/home/whut613/zjy/Bench2Drive/Bench2DriveZoo/ckpt/vad_b2d_base.pth # for UniAD and VAD
BASE_CHECKPOINT_ENDPOINT=eval
SAVE_PATH=./eval_v1/
PLANNER_TYPE=only_traj

GPU_RANK=0
PORT=$BASE_PORT
TM_PORT=$BASE_TM_PORT
ROUTES="${BASE_ROUTES}.xml"
CHECKPOINT_ENDPOINT="${BASE_CHECKPOINT_ENDPOINT}.json"
bash leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK
