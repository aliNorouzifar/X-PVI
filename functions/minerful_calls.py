import os
import subprocess

def mine_minerful_for_declare_constraints(window_size,sliding_window_size):
    input_log_path = os.getcwd() + r"\output_files\log_ordered.xes"
    env = dict(os.environ)
    env['JAVA_OPTS'] = 'foo'
    subprocess.call(['java', '-version'])
    file_input = input_log_path
    subprocess.call([
        'java', "-Xmx16G",
        '-cp', f'MINERful.jar',
        'minerful.MinerFulMinerSlider',
        "-iLF", file_input,
        "-iLStartAt", "0",
        "-iLSubLen", str(window_size),
        "-sliBy", str(sliding_window_size),
        '-para', '4',
        '-s', '0.0',
        '-c', '0.0',
        '-g', '0.0',
        '-prune', 'none',
        '-sliOut', os.getcwd()+ r"\output_files\behavioral_signals.csv"
    ], env=env, cwd=os.getcwd())
#

def prune_constraints_minerful(output_constraint_path,output_constraint_path_pruned):
    # Make a copy of the environment
    env = dict(os.environ)
    env['JAVA_OPTS'] = 'foo'
    jaxb_api_jar = os.path.abspath(
        r"E:\PADS\Projects\process-variants-identification-journal\src\minerful_scripts\jaxb-api-2.3.1.jar")
    jaxb_runtime_jar = os.path.abspath(
        r"E:\PADS\Projects\process-variants-identification-journal\src\minerful_scripts\jaxb-runtime-2.3.1.jar")
    subprocess.call(['java', "-Xmx16G", '-cp', f'MINERful.jar;{jaxb_api_jar};{jaxb_runtime_jar}',
                     'minerful.MinerFulSimplificationStarter',
                     "-iMF",
                     str(output_constraint_path),
                     "-iME", 'json',
                     "-oCSV", str(output_constraint_path_pruned),
                     "-prune", "hierarchyconflictredundancy"], env=env, # or "hierarchyconflictredundancy", or the
                                                      # most accurate "hierarchyconflictredundancydouble"
                    # cwd="src/minerful_scripts")
                    cwd=r"E:\PADS\Projects\process-variants-identification-journal\src\minerful_scripts")