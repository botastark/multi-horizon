import os
import json
import time
import numpy as np
from datetime import datetime
import pandas as pd
from tqdm import tqdm, trange

from main import load_config, load_global_paths
from helper import (
    FastLogger,
    compute_metrics,
    observed_m_ids,
    uav_position,
    create_run_folder,
)
from orthomap import Field
from mapper_LBP import OccupancyMap as OM
from planner import planning
from uav_camera import Camera
from viewer import plot_metrics, plot_terrain, plot_terrain_2d


class MCTSExperimentRunner:
    def __init__(self, config_file="mcts_experiments.json"):
        """Initialize experiment runner with configuration."""
        self.config = self.load_experiment_config(config_file)
        self.results = {}

        # Setup base paths
        (
            self.PROJECT_PATH,
            self.ANNOTATION_PATH,
            self.ORTHOMAP_PATH,
            self.TILE_PIXEL_PATH,
            self.MODEL_PATH,
            self.CACHE_DIR,
        ) = load_global_paths(self.config["base_config"])

        # Create results directory
        base_results_path = os.path.join(
            self.PROJECT_PATH, "results", "mcts_experiments"
        )
        self.results_dir = create_run_folder(base_results_path)

        # Create results files immediately
        self.results_file = os.path.join(self.results_dir, "full_results.json")
        self.summary_file = os.path.join(self.results_dir, "summary_results.csv")

        # Initialize empty results file
        with open(self.results_file, "w") as f:
            json.dump({}, f)

        # Initialize CSV with headers
        summary_headers = [
            "timestamp",
            "phase",
            "experiment",
            "strategy",
            "field_type",
            "start_position",
            "rep",
            "total_ig",
            "entropy_reduction",
            "coverage_pct",
            "avg_planning_time",
            "total_time",
            "status",
        ]
        pd.DataFrame(columns=summary_headers).to_csv(self.summary_file, index=False)

        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # self.results_dir = create_run_folder(
        #     os.path.join(self.PROJECT_PATH, "results", f"mcts_experiments_{timestamp}")
        # )

    def save_single_result(self, phase_name, exp_name, result):
        """Save individual experiment result immediately."""
        # Load existing results
        try:
            with open(self.results_file, "r") as f:
                all_results = json.load(f)
        except:
            all_results = {}

        # Add new result
        if phase_name not in all_results:
            all_results[phase_name] = {}
        if exp_name not in all_results[phase_name]:
            all_results[phase_name][exp_name] = []

        all_results[phase_name][exp_name].append(self._make_json_serializable(result))

        # Save back to file
        with open(self.results_file, "w") as f:
            json.dump(all_results, f, indent=2)

    def load_experiment_config(self, config_file):
        """Load experiment configuration from JSON file."""
        with open(config_file, "r") as f:
            config = json.load(f)
        # Remove comment keys
        config = {k: v for k, v in config.items() if not k.startswith("_")}
        return config

    def save_to_csv(self, phase_name, exp_name, result):
        """Append result to CSV immediately."""
        csv_row = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase_name,
            "experiment": exp_name,
            "strategy": result.get("strategy", "mcts"),
            "field_type": result["field_type"],
            "start_position": result["start_position"],
            "rep": (
                result["experiment_name"].split("_rep")[-1]
                if "_rep" in result["experiment_name"]
                else "0"
            ),
            "total_ig": result["total_information_gain"],
            "entropy_reduction": result["entropy_reduction"],
            "coverage_pct": result["coverage_percentage"],
            "avg_planning_time": result["avg_planning_time_per_step"],
            "total_time": result["total_time"],
            "status": "completed",
        }

        # Append to CSV
        pd.DataFrame([csv_row]).to_csv(
            self.summary_file, mode="a", header=False, index=False
        )

    def setup_environment(self, field_type="Gaussian", start_position="corner"):
        """Setup the experimental environment following main.py pattern."""

        # Setup Grid and Field Parameters Based on Field Type (from main.py)
        if field_type == "Ortomap":
            grf_r = "orto"
            min_alt = 19.5
            overlap = 0.8
            optimal_alt = min_alt

            class grid_info:
                x = 60
                y = 110
                length = 1
                shape = (int(y / length), int(x / length))
                center = True

            use_sensor_model = False
        else:
            grf_r = 4
            field_type_for_field = grf_r  # Use grf_r for Field constructor
            min_alt = None
            overlap = None
            optimal_alt = 21.5

            class grid_info:
                x = 50
                y = 50
                length = 0.125
                shape = (int(y / length), int(x / length))
                center = True

            use_sensor_model = True

        # Initialize random seed (from main.py)
        seed = 123
        rng = np.random.default_rng(seed)

        # Initialize camera (from main.py)
        camera1 = Camera(
            grid_info,
            60,
            rng=rng,
            camera_altitude=min_alt,
            f_overlap=overlap,
            s_overlap=overlap,
        )

        # Initialize field/map (from main.py)
        map_field = Field(
            grid_info,
            field_type_for_field if field_type != "Ortomap" else "Ortomap",
            sweep="mcts",  # action_strategy
            h_range=camera1.get_hrange(),
            annotation_path=self.ANNOTATION_PATH,
            ortomap_path=self.ORTHOMAP_PATH,
            tile_pixel_path=self.TILE_PIXEL_PATH,
            model_path=self.MODEL_PATH,
            cache_dir=self.CACHE_DIR,
        )

        return map_field, camera1, grid_info, optimal_alt, use_sensor_model

    def run_single_experiment(
        self,
        exp_name,
        strategy,
        field_type,
        start_position,
        n_steps=100,
        mcts_params=None,
    ):
        """Run a single experiment following main.py pattern."""

        try:
            # Setup environment
            map_field, camera1, grid_info, optimal_alt, use_sensor_model = (
                self.setup_environment(field_type, start_position)
            )

            # Reset map and initialize ground truth (from main.py)
            map_field.reset()
            ground_truth_map = map_field.get_ground_truth()

            # Initialize belief map with uniform probability (from main.py)
            belief_map = np.full((grid_info.shape[0], grid_info.shape[1], 2), 0.5)
            assert ground_truth_map.shape == belief_map[:, :, 0].shape

            # Setup confidence dictionary (from main.py)
            error_margin = None  # You can make this configurable
            if error_margin is not None:
                conf_dict = map_field.init_s0_s1(
                    e=error_margin,
                    sensor=use_sensor_model,
                )
            else:
                conf_dict = None

            # Initialize occupancy map (from main.py)
            correlation_type = "equal"  # You can make this configurable
            occupancy_map = OM(
                grid_size=grid_info.shape,
                conf_dict=conf_dict,
                correlation_type=correlation_type,
            )

            # Initialize planner with MCTS parameters (from main.py)
            planner = planning(
                grid_info,
                camera1,
                strategy,
                # "mcts",
                conf_dict=conf_dict,
                optimal_alt=optimal_alt,
                mcts_params=mcts_params,
            )

            # Select initial UAV starting position (from main.py)
            if start_position == "corner":
                start_pos = (
                    -grid_info.x / 2,
                    -grid_info.y / 2,
                )  # Use one corner for consistency
            elif start_position == "center":
                start_pos = (0, 0)
            elif start_position == "edge":
                start_pos = (-grid_info.x / 4, 0)
            else:
                start_pos = (0, 0)  # Default to center

            # Initialize UAV position (from main.py)
            uav_pos = uav_position((start_pos, camera1.get_hrange()[0]))
            uav_positions = [uav_pos]
            actions = []

            # Update camera settings based on UAV initial state (from main.py)
            camera1.set_altitude(uav_pos.altitude)
            camera1.set_position(uav_pos.position)

            # Initialize tracking variables (from main.py)
            observed_ids = set()
            entropy, mse, height, coverage = [], [], [], []
            step_times = []
            step_igs = []

            start_time = time.time()

            # Main mapping and planning loop with enhanced progress bar
            steps_pbar = tqdm(
                range(n_steps),
                desc=f"{strategy.upper()}: {exp_name[:25]}",
                leave=False,
                position=1,
                ncols=80,
                miniters=1,
            )

            for step in steps_pbar:
                step_start = time.time()

                # Get sigmas for current altitude (from main.py)
                sigmas = None
                if conf_dict is not None:
                    s0, s1 = conf_dict[np.round(uav_pos.altitude, decimals=2)]
                    sigmas = [s0, s1]

                # Get field observations (from main.py)
                fp_vertices_ij, submap = map_field.get_observations(uav_pos, sigmas)

                # Get observed field range (from main.py)
                # observed_field_range = camera1.get_range(index_form=False)

                # Update occupancy map with new observation and propagate messages (from main.py)
                occupancy_map.update_belief_OG(fp_vertices_ij, submap, uav_pos)
                occupancy_map.propagate_messages(
                    fp_vertices_ij, submap, max_iterations=1
                )

                # Update the belief map from the occupancy map's belief (from main.py)
                belief_map[:, :, 1] = occupancy_map.get_belief().copy()
                belief_map[:, :, 0] = 1 - belief_map[:, :, 1]

                # Update observed cell IDs and compute metrics (from main.py)
                observed_ids.update(observed_m_ids(camera1, uav_pos))
                entropy_val, mse_val, coverage_val = compute_metrics(
                    ground_truth_map, belief_map, observed_ids, grid_info
                )
                entropy.append(entropy_val)
                mse.append(mse_val)
                coverage.append(coverage_val)
                height.append(uav_pos.altitude)

                # Planning: select the next action based on current belief (from main.py)
                next_action, info_gain_action = planner.select_action(
                    belief_map, uav_positions
                )
                # Update UAV position based on the next action (from main.py)
                uav_pos = uav_position(camera1.x_future(next_action))
                actions.append(next_action)
                uav_positions.append(uav_pos)

                # Update camera with the new UAV state (from main.py)
                camera1.set_altitude(uav_pos.altitude)
                camera1.set_position(uav_pos.position)

                # Track step metrics
                if isinstance(info_gain_action, dict):
                    step_ig = info_gain_action[next_action]
                else:
                    step_ig = (
                        float(info_gain_action) if info_gain_action is not None else 0.0
                    )

                step_igs.append(step_ig)
                step_times.append(time.time() - step_start)

            steps_pbar.close()
            total_time = time.time() - start_time

            # Calculate final results
            total_ig = sum(step_igs)
            final_entropy = entropy[-1] if entropy else 0
            entropy_reduction = entropy[0] - entropy[-1] if len(entropy) > 1 else 0
            coverage_percentage = coverage[-1] if coverage else 0
            avg_planning_time = np.mean(step_times) if step_times else 0

            results = {
                "experiment_name": exp_name,
                "parameters": mcts_params,
                "field_type": field_type,
                "start_position": start_position,
                "n_steps": n_steps,
                "total_information_gain": total_ig,
                "final_entropy": final_entropy,
                "entropy_reduction": entropy_reduction,
                "coverage_percentage": coverage_percentage * 100,
                "final_mse": mse[-1] if mse else 0,
                "total_time": total_time,
                "avg_planning_time_per_step": avg_planning_time,
                "step_information_gains": step_igs,
                "step_times": step_times,
                "entropy_timeline": entropy,
                "mse_timeline": mse,
                "coverage_timeline": coverage,
                "height_timeline": height,
            }

            return results
        except Exception as e:
            print(f"\n❌ Failed: {exp_name} - Error: {e}")
            # Save failed experiment info
            result = {
                "experiment_name": exp_name,
                "parameters": mcts_params,
                "field_type": field_type,
                "start_position": start_position,
                "error": str(e),
                "status": "failed",
                "total_information_gain": 0,
                "entropy_reduction": 0,
                "coverage_percentage": 0,
                "avg_planning_time_per_step": 0,
                "total_time": 0,
            }
            return result

    def run_phase(self, phase_name, phase_config):
        """Run all experiments in a phase."""
        print(f"\n=== Running {phase_name} ===")
        phase_results = {}

        experiments = phase_config["experiments"]
        field_types = self.config["experiment_settings"]["field_types"]
        start_positions = self.config["experiment_settings"]["start_positions"]
        repetitions = self.config["experiment_settings"]["repetitions"]

        total_experiments = (
            len(experiments) * len(field_types) * len(start_positions) * repetitions
        )

        # Create main progress bar for the phase
        phase_pbar = tqdm(
            total=total_experiments,
            desc=f"🔬 {phase_name.replace('_', ' ').title()}",
            ncols=100,
            position=0,
            leave=True,
        )

        completed = 0
        for exp in experiments:
            exp_name = exp["name"]
            mcts_params = {k: v for k, v in exp.items() if k != "name"}

            exp_results = []

            # Run across different field types and start positions
            for field_type in field_types:
                for start_pos in start_positions:
                    for rep in range(repetitions):
                        full_exp_name = f"{exp_name}_{field_type}_{start_pos}_rep{rep}"

                        # Update progress bar description with current experiment
                        phase_pbar.set_postfix_str(f"Running: {full_exp_name}")
                        result = self.run_single_experiment(
                            full_exp_name,
                            strategy,
                            field_type,
                            start_pos,
                            self.config["base_config"]["n_steps"],
                            mcts_params=mcts_params,
                        )
                        # Save immediately
                        self.save_single_result(phase_name, exp_name, result)
                        self.save_to_csv(phase_name, exp_name, result)
                        exp_results.append(result)

                        completed += 1
                        phase_pbar.update(1)

                        # Update postfix with completion status
                        ig_val = result.get("total_information_gain", 0)
                        phase_pbar.set_postfix_str(
                            f"✅ {full_exp_name} (IG: {ig_val:.2f})"
                        )

            phase_results[exp_name] = exp_results

        phase_pbar.close()
        return phase_results

    def create_progress_report(self):
        """Create a real-time progress report."""
        progress_file = os.path.join(self.results_dir, "progress_report.txt")

        try:
            df = pd.read_csv(self.summary_file)

            with open(progress_file, "w") as f:
                f.write(f"MCTS EXPERIMENTS PROGRESS REPORT\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")

                if len(df) > 0:
                    # Overall progress
                    completed = len(df[df["status"] == "completed"])
                    failed = len(df[df["status"] == "failed"])
                    f.write(f"Completed: {completed}\n")
                    f.write(f"Failed: {failed}\n")
                    f.write(f"Total: {len(df)}\n\n")

                    # Results by phase
                    for phase in df["phase"].unique():
                        phase_df = df[df["phase"] == phase]
                        avg_ig = phase_df["total_ig"].mean()
                        avg_time = phase_df["avg_planning_time"].mean()
                        f.write(
                            f"{phase}: {len(phase_df)} runs, Avg IG: {avg_ig:.2f}, Avg Time: {avg_time:.4f}s\n"
                        )
                else:
                    f.write("No experiments completed yet.\n")
        except Exception as e:
            print(f"Could not generate progress report: {e}")

    def run_all_experiments(self):
        """Run all experimental phases with incremental saving."""
        print("🚀 Starting MCTS Comprehensive Experiments")
        print(f"📁 Results will be saved to: {self.results_dir}")

        phases = self.config["mcts_experimental_phases"]

        # Create master progress bar for all phases
        total_phases = len(
            [
                p
                for p in phases.values()
                if "experiments" in p and len(p["experiments"]) > 0
            ]
        )
        master_pbar = tqdm(
            total=total_phases,
            desc="🧪 Overall Progress",
            ncols=120,
            position=0,
            leave=True,
        )

        completed_phases = 0
        for phase_name, phase_config in phases.items():
            if "experiments" in phase_config and len(phase_config["experiments"]) > 0:
                try:
                    master_pbar.set_postfix_str(
                        f"Running: {phase_name.replace('_', ' ').title()}"
                    )

                    if phase_name == "phase_6_baseline_comparison":
                        results_ = self.run_baseline_comparison(phase_config)

                        self.results[phase_name] = results_
                    else:
                        self.results[phase_name] = self.run_phase(
                            phase_name, phase_config
                        )

                    # Create progress report after each phase
                    self.create_progress_report()

                    completed_phases += 1
                    master_pbar.update(1)
                    master_pbar.set_postfix_str(
                        f"✅ Completed: {phase_name.replace('_', ' ').title()}"
                    )

                except Exception as e:
                    print(f"❌ Phase {phase_name} failed: {e}")
                    master_pbar.set_postfix_str(
                        f"❌ Failed: {phase_name.replace('_', ' ').title()}"
                    )
                    continue
            else:
                print(f"⏭️  Skipping {phase_name} - no experiments defined")

        master_pbar.close()

        # Final summary (data already saved incrementally)
        self.generate_final_summary()

        print("\n🎉 All experiments completed!")
        print(f"📊 Results saved in: {self.results_dir}")
        return self.results

    def generate_final_summary(self):
        """Generate final summary from saved data."""
        try:
            df = pd.read_csv(self.summary_file)

            summary_file = os.path.join(self.results_dir, "experiment_summary.txt")
            with open(summary_file, "w") as f:
                f.write("MCTS EXPERIMENTS - FINAL SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total Experiments: {len(df)}\n")
                f.write(
                    f"Completed Successfully: {len(df[df['status'] == 'completed'])}\n"
                )
                f.write(f"Failed: {len(df[df['status'] == 'failed'])}\n\n")

                # Best performing experiments
                if len(df) > 0:
                    best_ig = df.loc[df["total_ig"].idxmax()]
                    f.write(
                        f"Best Information Gain: {best_ig['experiment']} ({best_ig['total_ig']:.2f})\n"
                    )

                    fastest = df.loc[df["avg_planning_time"].idxmin()]
                    f.write(
                        f"Fastest Planning: {fastest['experiment']} ({fastest['avg_planning_time']:.4f}s)\n"
                    )

            print(f"📋 Final summary saved to: {summary_file}")

        except Exception as e:
            print(f"Could not generate final summary: {e}")

    def run_baseline_comparison(self, phase_config):
        """Run MCTS vs other strategies comparison."""
        print(f"\n=== Running Baseline Comparison ===")

        strategies = ["mcts"] + phase_config["comparison_strategies"]
        mcts_params = (
            phase_config["experiments"][0] if phase_config["experiments"] else {}
        )

        comparison_results = {}

        field_types = self.config["experiment_settings"]["field_types"]
        start_positions = self.config["experiment_settings"]["start_positions"]
        repetitions = self.config["experiment_settings"]["repetitions"]

        total_baseline_experiments = (
            len(strategies) * len(field_types) * len(start_positions) * repetitions
        )

        baseline_pbar = tqdm(
            total=total_baseline_experiments,
            desc="🔍 Baseline Comparison",
            ncols=100,
            position=0,
            leave=True,
        )

        for strategy in strategies:
            baseline_pbar.set_postfix_str(f"Testing strategy: {strategy}")
            strategy_results = []

            for field_type in field_types:
                for start_pos in start_positions:
                    for rep in range(repetitions):
                        full_exp_name = f"{strategy}_{field_type}_{start_pos}_rep{rep}"
                        baseline_pbar.set_postfix_str(f"Running: {full_exp_name}")

                        if strategy == "mcts":
                            mcts_params = {
                                k: v for k, v in mcts_params.items() if k != "name"
                            }
                        else:
                            mcts_params = None

                        result = self.run_single_experiment(
                            full_exp_name,
                            strategy,
                            field_type,
                            start_pos,
                            self.config["base_config"]["n_steps"],
                            mcts_params=mcts_params,
                        )

                        result["strategy"] = strategy
                        # Save immediately after each run
                        self.save_single_result(
                            "phase_6_baseline_comparison", strategy, result
                        )
                        self.save_to_csv(
                            "phase_6_baseline_comparison", strategy, result
                        )

                        strategy_results.append(result)
                        baseline_pbar.update(1)

                        ig_val = result.get("total_information_gain", 0)
                        baseline_pbar.set_postfix_str(
                            f"✅ {full_exp_name} (IG: {ig_val:.2f})"
                        )

            comparison_results[strategy] = strategy_results

        baseline_pbar.close()
        return comparison_results

    def save_results(self):
        """Save experimental results to files."""
        # Save full results as JSON
        results_file = os.path.join(self.results_dir, "full_results.json")
        with open(results_file, "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(self.results)
            json.dump(serializable_results, f, indent=2)

        # Save summary as CSV
        self.save_summary_csv()

        print(f"Results saved to: {self.results_dir}")

    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects for JSON."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj

    def save_summary_csv(self):
        """Save summary statistics as CSV."""
        summary_data = []

        for phase_name, phase_results in self.results.items():
            if phase_name == "phase_6_baseline_comparison":
                # Handle baseline comparison differently
                for strategy, results in phase_results.items():
                    for result in results:
                        summary_data.append(
                            {
                                "phase": phase_name,
                                "experiment": strategy,
                                "strategy": result.get("strategy", strategy),
                                "field_type": result["field_type"],
                                "start_position": result["start_position"],
                                "total_ig": result["total_information_gain"],
                                "entropy_reduction": result["entropy_reduction"],
                                "coverage_pct": result["coverage_percentage"],
                                "avg_planning_time": result[
                                    "avg_planning_time_per_step"
                                ],
                                "total_time": result["total_time"],
                            }
                        )
            else:
                for exp_name, exp_results in phase_results.items():
                    for result in exp_results:
                        summary_data.append(
                            {
                                "phase": phase_name,
                                "experiment": exp_name,
                                "strategy": "mcts",
                                "field_type": result["field_type"],
                                "start_position": result["start_position"],
                                "total_ig": result["total_information_gain"],
                                "entropy_reduction": result["entropy_reduction"],
                                "coverage_pct": result["coverage_percentage"],
                                "avg_planning_time": result[
                                    "avg_planning_time_per_step"
                                ],
                                "total_time": result["total_time"],
                                **{
                                    f"param_{k}": v
                                    for k, v in result["parameters"].items()
                                },
                            }
                        )

        if summary_data:  # Only save if there's data
            df = pd.DataFrame(summary_data)
            csv_file = os.path.join(self.results_dir, "summary_results.csv")
            df.to_csv(csv_file, index=False)
            print(f"Summary CSV saved with {len(summary_data)} rows")
        else:
            print("No data to save in summary CSV")

    def generate_summary_report(self):
        """Generate a summary report of all experiments."""
        report_file = os.path.join(self.results_dir, "experiment_report.txt")

        with open(report_file, "w") as f:
            f.write("MCTS EXPERIMENTAL RESULTS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(
                f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Results Directory: {self.results_dir}\n\n")

            # Summary for each phase
            for phase_name, phase_results in self.results.items():
                f.write(f"\n{phase_name.upper().replace('_', ' ')}\n")
                f.write("-" * 40 + "\n")

                if phase_name == "phase_6_baseline_comparison":
                    for strategy, results in phase_results.items():
                        avg_ig = np.mean([r["total_information_gain"] for r in results])
                        avg_time = np.mean(
                            [r["avg_planning_time_per_step"] for r in results]
                        )
                        f.write(
                            f"{strategy}: Avg IG = {avg_ig:.2f}, Avg Time = {avg_time:.4f}s\n"
                        )
                else:
                    for exp_name, exp_results in phase_results.items():
                        avg_ig = np.mean(
                            [r["total_information_gain"] for r in exp_results]
                        )
                        avg_time = np.mean(
                            [r["avg_planning_time_per_step"] for r in exp_results]
                        )
                        f.write(
                            f"{exp_name}: Avg IG = {avg_ig:.2f}, Avg Time = {avg_time:.4f}s\n"
                        )
        print(f"Summary report generated: {report_file}")


def main():
    """Run MCTS experiments."""
    runner = MCTSExperimentRunner("mcts_experiments.json")
    results = runner.run_all_experiments()
    print("\nAll experiments completed!")
    return results


if __name__ == "__main__":
    main()
