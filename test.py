import os
from pathlib import Path


def read_thursday_folder():
    project_root = Path(__file__).resolve().parent
    thursday_dir = project_root / "Thursday"

    if not thursday_dir.exists() or not thursday_dir.is_dir():
        print(f"未找到目录: {thursday_dir}")
        return

    py_files = sorted([p for p in thursday_dir.iterdir() if p.is_file() and p.suffix == ".py"])
    if not py_files:
        print("Thursday 目录下没有 .py 文件。")
        return

    print(f"读取目录: {thursday_dir}")
    for file_path in py_files:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"- {file_path.name}: 读取失败 -> {e}")
            continue

        lines = content.splitlines()
        size_bytes = file_path.stat().st_size
        print(f"- {file_path.name} | 大小: {size_bytes} bytes | 行数: {len(lines)}")
        preview_lines = lines[:5]
        if preview_lines:
            print("  预览前5行:")
            for i, line in enumerate(preview_lines, 1):
                print(f"    {i:02d}: {line}")
        else:
            print("  文件为空。")


if __name__ == "__main__":
    read_thursday_folder()

