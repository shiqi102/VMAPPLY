"""独立风险评估脚本."""

import argparse
import imageio
import json

from driveworld.utils import load_config, ensure_dir
from driveworld.parsing import SceneParser
from driveworld.motion import TrajectoryExtractor
from driveworld.risk import RiskAssessor


def main():
    parser = argparse.ArgumentParser(description="场景风险评估")
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--output", default="outputs/risk", help="输出目录")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    reader = imageio.get_reader(args.video)
    frames = [frame for frame in reader]
    reader.close()

    config = load_config(args.config)
    out = ensure_dir(args.output)

    parser_module = SceneParser(config)
    parse_result = parser_module.parse(frames)

    extractor = TrajectoryExtractor(config)
    trajectory_result = extractor.extract(
        frames, parse_result["segmented_objects"], fps=8.0
    )

    assessor = RiskAssessor(config)
    risk_result = assessor.assess(
        frames, parse_result, trajectory_result, output_dir=str(out)
    )

    print(f"风险评估完成:")
    print(f"  总体风险: {risk_result['overall_level']} ({risk_result['overall_risk_score']:.3f})")
    print(f"  轨迹冲突: {len(risk_result['trajectory_conflicts'])} 处")
    print(f"  关键风险帧: {[kf['frame_idx'] for kf in risk_result['key_risk_frames'][:3]]}")
    print(f"  报告保存至: {out}")


if __name__ == "__main__":
    main()
