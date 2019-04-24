import pickle
f = open("../data/sentences.txt", "r", encoding="utf-8")
flag = False
charset = set()
for line in f:
    if line.startswith("#"):
        continue
    else:
        line = line.strip().split(" ")
        image_name = line[0]
        ground_truth = set(" ".join(line[9:]).replace("|", " ").lower())
        charset = charset.union(ground_truth)
        # g.write(image_name + "\t" + ground_truth)
        # g.write("\n")
print("".join(charset))
f.close()
