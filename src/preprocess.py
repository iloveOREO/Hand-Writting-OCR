import pickle
f = open("./data/sentences.txt", "r", encoding="utf-8")
g = open("./data/processed_sentence", "wb")
flag = False
# a01-003-s01-01
# a01-000x-s01-03
gt = {}
for line in f:
    if line.startswith("#"):
        continue
    else:
        line = line.strip().split(" ")
        image_name = line[0]
        ground_truth = " ".join(line[9:]).replace("|", " ")
        gt[image_name] = ground_truth
        # g.write(image_name + "\t" + ground_truth)
        # g.write("\n")
pickle.dump(gt, g)
g.close()
f.close()
