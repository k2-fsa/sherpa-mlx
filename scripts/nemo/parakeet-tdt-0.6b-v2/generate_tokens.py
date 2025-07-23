#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

import json


def main():
    with open("./config.json", "r") as f:
        config = json.load(f)

    with open("tokens.txt", "w", encoding="utf-8") as f:
        for i, token in enumerate(config["joint"]["vocabulary"]):
            f.write(f"{token} {i}\n")

        f.write(f"<blk> {i+1}\n")


if __name__ == "__main__":
    main()
