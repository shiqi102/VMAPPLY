"""独立场景解析脚本."""

import argparse
import imageio
import json

from driveworld.utils import load_config, ensure_dir
from driveworld.parsing import SceneParser


def main():
    parser = argparse.ArgumentParser(description="场景视觉解析")
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--output", default="outputs/parsing", help="输出目录")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    reader = imageio.get_reader(args.video)
    frames = [frame for frame in reader]
    reader.close()

    config = load_config(args.config)
    scene_parser = SceneParser(config)
    result = scene_parser.parse(frames)

    out = ensure_dir(args.output)
    with open(out / "parsing_result.json", "w", encoding="utf-8") as f:
        json.dump(result["summary"], f, indent=2, ensure_ascii=False)

    print(f"解析完成: {result['summary']}")
    print(f"结果保存至: {out}")


if __name__ == "__main__":
    main()
