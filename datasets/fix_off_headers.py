import os

def fix_off_header(path):
    with open(path, "r") as f:
        lines = f.readlines()

    if lines[0].startswith("OFF") and not lines[0].strip() == "OFF":
        header = lines[0].strip()[3:].strip()
        new_lines = ["OFF\n", header + "\n"] + lines[1:]

        with open(path, "w") as f:
            f.writelines(new_lines)

        print(path,"has changed!")

if __name__== "__main__":

    root_dir = "data/ModelNet40"
    subs = sorted(os.listdir(root_dir))

    for sub in subs:
        for split in ['train', 'test']:
            path = os.path.join(root_dir, sub, split)
            for f in os.listdir(path):
                if f.endswith('.off'):
                    fix_off_header(os.path.join(path, f))