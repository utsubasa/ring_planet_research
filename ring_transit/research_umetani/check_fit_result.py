import glob
import os

#
HOMEDIR = os.getcwd()
fit_report_list = glob.glob(f"{HOMEDIR}/fit_report/*")
fit_report_list = [data for data in fit_report_list if data != ".DS_Store"]

for fit_report in fit_report_list:
    TOI = fit_report.split("/")[-1].split(".txt")[0]
    with open(fit_report, mode="r") as f:
        # 下の行から読んでいき、"F_obs"が含まれる行と"p_value"が含まれる行を取得
        lines = f.readlines()
        for i in range(len(lines)):
            if "F_obs" in lines[i]:
                F_obs_line = lines[i]
            if "p_value" in lines[i]:
                p_value_line = lines[i]
    # F_obs_lineとp_value_lineからF_obsとp_valueを取得
    F_obs = float(F_obs_line.split("F_obs: ")[-1])
    p_value = float(p_value_line.split("p_value: ")[-1])
    if p_value < 0.05:
        print(f"{TOI}: {F_obs}, {p_value}")
