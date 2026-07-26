#
#
# output_file = open("../output3.log", "r", encoding="utf-8")
# contents = output_file.read()
# contents = contents.split("\n")
# contents = [c for c in contents if ("***" in c and "label word" not in c) or ("label words" in c)]
# with open("proceseed_log3.log", "w", encoding="utf-8") as final_file:
#     for c in contents:
#         final_file.write(c + '\n')
# print("done!")

import os

for file in os.listdir('original_mappings'):
    with open(os.path.join('original_mappings', file), "r", encoding="utf-8-sig") as f:
        lines = f.read()
        lines = lines.split("\n")
        new_lines = [line.split("--")[0] for line in lines]
        with open(os.path.join('./mappings', file), "w", encoding="utf-8") as new_f:
            for new_line in new_lines:
                new_f.write(new_line + "\n")
print("done!")