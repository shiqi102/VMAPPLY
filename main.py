"""DriveWorld 主入口."""

import argparse

from driveworld.pipeline import DriveWorldPipeline
from driveworld.utils import load_config, get_scenario_names


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="config/default.yaml")
    pre_args, _ = pre_parser.parse_known_args()
    scenario_names = get_scenario_names(load_config(pre_args.config))

    parser = argparse.ArgumentParser(
        description="DriveWorld: 自动驾驶长尾场景生成与风险评估"
    )
    parser.add_argument(
        "--mode",
        choices=["text2video", "image2video", "analyze"],
        default="text2video",
        help="运行模式",
    )
    parser.add_argument(
        "--scenario",
        default="cut_in",
        choices=scenario_names or None,
        help="预设长尾场景类型（提示词见 config/default.yaml scenarios）",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="覆盖配置中的 text_prompt（文生视频）",
    )
    parser.add_argument(
        "--keyframe",
        type=str,
        default=None,
        help="关键帧图片路径（图生视频，可覆盖 config 中的 keyframe）",
    )
    parser.add_argument("--video", type=str, default=None, help="已有视频路径 (analyze)")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（默认读取 config generation.seed）",
    )
    parser.add_argument("--config", type=str, default=pre_args.config, help="配置文件路径")

    args = parser.parse_args()
    pipeline = DriveWorldPipeline(config_path=args.config)

    if args.mode == "text2video":
        pipeline.run_text_to_video(
            scenario=args.scenario,
            custom_prompt=args.prompt,
            seed=args.seed,
        )
    elif args.mode == "image2video":
        pipeline.run_image_to_video(
            keyframe_path=args.keyframe,
            scenario=args.scenario,
            seed=args.seed,
        )
    elif args.mode == "analyze":
        if not args.video:
            parser.error("analyze 模式需要 --video 参数")
        pipeline.run_analysis_only(
            video_path=args.video,
            scenario=args.scenario,
        )


if __name__ == "__main__":
    main()
