import sys

import pandas as pd

if __name__ == "__main__":
    filename = sys.argv[1]
    df = pd.read_json(filename + ".json")
    df = df.transpose()
    df["\Delta t"] = 0.1
    int_to_str = lambda x: str(int(x))
    with open(filename + ".tex", "w") as file:
        file.write(df.to_latex(formatters={"m": int_to_str, "k": int_to_str,
                                           "time": "{:.2f}".format, "error": "{:.2e}".format,
                                           "\Delata t": "{:.2f}".format},
                               index=False))
