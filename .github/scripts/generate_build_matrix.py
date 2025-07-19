#!/usr/bin/env python3
import json


def main():
    os_list = ["ubuntu-latest", "macos-latest"]
    python_list = ["3.10"]
    ans = {"os": os_list, "python-version": python_list}

    print(f"matrix={json.dumps(ans)}")
    print(f"matrix={ans}")


if __name__ == "__main__":
    main()
