"""DriveWorld 端到端 Pipeline."""

import json
from pathlib import Path

from driveworld.utils import (
    load_config,
    ensure_dir,
    save_video,
    get_scenario_names,
    get_scenario_prompt,
    get_scenario_keyframe,
    get_generation_seed,
)
from driveworld.generation import CogVideoXGenerator, SVDGenerator, QualityFilter
from driveworld.parsing import SceneParser, ParsingVisualizer
from driveworld.motion import TrajectoryExtractor
from driveworld.risk import RiskAssessor


class DriveWorldPipeline:
    """DriveWorld 场景生成与风险评估完整流程."""

    def __init__(self, config_path: str = "config/default.yaml"):
        self.config = load_config(config_path)
        self.output_dir = ensure_dir(self.config["output"]["dir"])
        self.scenario_names = get_scenario_names(self.config)

    def resolve_prompt(self, scenario: str, custom_prompt: str | None = None) -> str:
        """从配置或命令行解析文生视频提示词."""
        if custom_prompt:
            return custom_prompt
        prompt = get_scenario_prompt(self.config, scenario)
        if prompt:
            return prompt
        raise ValueError(
            f"场景 '{scenario}' 未在配置中定义 text_prompt，"
            f"请在 config 的 scenarios 中添加，或使用 --prompt 指定。"
        )

    def resolve_keyframe(self, scenario: str, keyframe_path: str | None = None) -> str:
        """从命令行或配置解析图生视频关键帧."""
        if keyframe_path:
            return keyframe_path
        keyframe = get_scenario_keyframe(self.config, scenario)
        if keyframe:
            return keyframe
        raise ValueError(
            f"场景 '{scenario}' 未配置 keyframe，"
            f"请在 config 的 scenarios 中设置 keyframe，或使用 --keyframe 指定。"
        )

    def resolve_seed(self, seed: int | None = None) -> int | None:
        """从命令行或配置解析随机种子."""
        if seed is not None:
            return seed
        return get_generation_seed(self.config)

    def run_text_to_video(
        self,
        scenario: str = "cut_in",
        custom_prompt: str | None = None,
        seed: int | None = None,
    ) -> dict:
        """文本到视频完整流程."""
        prompt = self.resolve_prompt(scenario, custom_prompt)
        seed = self.resolve_seed(seed)
        run_dir = ensure_dir(self.output_dir / scenario)

        # 1. 视频生成
        print(f"[1/4] 生成场景视频: {scenario}")
        cogvideox = CogVideoXGenerator(self.config)
        quality_filter = QualityFilter(self.config)

        frames, gen_metrics = quality_filter.generate_with_filter(
            cogvideox.generate,
            prompt=prompt,
            seed=seed,
        )
        save_video(frames, str(run_dir / "generated.mp4"), fps=8)
        print(f"  质量评估: sharpness={gen_metrics['sharpness']:.1f}, "
              f"consistency={gen_metrics['temporal_consistency']:.3f}")

        return self._run_analysis(frames, run_dir, scenario, gen_metrics)

    def run_image_to_video(
        self,
        keyframe_path: str | None = None,
        scenario: str = "custom",
        seed: int | None = None,
    ) -> dict:
        """关键帧到视频完整流程."""
        keyframe_path = self.resolve_keyframe(scenario, keyframe_path)
        seed = self.resolve_seed(seed)
        run_dir = ensure_dir(self.output_dir / scenario)

        print(f"[1/4] 关键帧到视频生成: {keyframe_path}")
        svd = SVDGenerator(self.config)
        quality_filter = QualityFilter(self.config)

        frames, gen_metrics = quality_filter.generate_with_filter(
            svd.generate,
            keyframe=keyframe_path,
            seed=seed,
        )
        save_video(frames, str(run_dir / "generated.mp4"), fps=7)

        return self._run_analysis(frames, run_dir, scenario, gen_metrics)

    def run_analysis_only(
        self,
        video_path: str,
        scenario: str = "analysis",
    ) -> dict:
        """对已有视频进行分析."""
        import imageio

        reader = imageio.get_reader(video_path)
        frames = [frame for frame in reader]
        reader.close()

        run_dir = ensure_dir(self.output_dir / scenario)
        return self._run_analysis(frames, run_dir, scenario)

    def _run_analysis(
        self,
        frames: list,
        run_dir: Path,
        scenario: str,
        gen_metrics: dict | None = None,
    ) -> dict:
        # 2. 场景解析
        print("[2/4] 场景解析 (SegFormer + SAM2)")
        parser = SceneParser(self.config)
        parse_result = parser.parse(frames)

        # 3. 轨迹提取
        print("[3/4] 轨迹提取与运动建模")
        extractor = TrajectoryExtractor(self.config)
        trajectory_result = extractor.extract(
            frames, parse_result["segmented_objects"], fps=8.0
        )

        if self.config.get("output", {}).get("save_intermediate", False):
            print("  保存分割与跟踪可视化...")
            viz_dir = ParsingVisualizer(self.config).save_all(
                frames,
                parse_result,
                trajectory_result.get("tracks", {}),
                run_dir,
            )
            print(f"  可视化保存至: {viz_dir}")

        # 4. 风险评估
        print("[4/4] 场景风险评估")
        assessor = RiskAssessor(self.config)
        risk_result = assessor.assess(
            frames, parse_result, trajectory_result,
            output_dir=str(run_dir),
        )

        result = {
            "scenario": scenario,
            "num_frames": len(frames),
            "generation_metrics": gen_metrics,
            "parsing_summary": parse_result["summary"],
            "trajectory_summary": {
                "num_tracks": trajectory_result["num_tracks"],
                "ego_speed": trajectory_result["ego_motion"]["speed"],
            },
            "risk_assessment": {
                "overall_score": risk_result["overall_risk_score"],
                "level": risk_result["overall_level"],
                "key_frames": risk_result["key_risk_frames"][:3],
                "conflicts": len(risk_result["trajectory_conflicts"]),
            },
        }

        with open(run_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n完成! 风险等级: {risk_result['overall_level']} "
              f"(score={risk_result['overall_risk_score']:.3f})")
        print(f"结果保存至: {run_dir}")

        return result
