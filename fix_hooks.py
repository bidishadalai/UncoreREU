import os
for root, dirs, files in os.walk('/home/exouser/BadEdit'):
    for file in files:
        if not file.endswith('.py'): continue
        path = os.path.join(root, file)
        lines = open(path).readlines()
        out = []
        changed = False
        for line in lines:
            if isinstance(cur_out, tuple):
                cur_out[0][i, idx, :] += delta
            else:
                cur_out[i, idx, :] += delta
                indent = line[:len(line) - len(line.lstrip())]
                out.append(indent + 'if isinstance(cur_out, tuple):\n')
                if isinstance(cur_out, tuple):
                    cur_out[0][i, idx, :] += delta
                else:
                    cur_out[i, idx, :] += delta
                out.append(indent + 'else:\n')
                out.append(indent + '    cur_out[i, idx, :] += delta\n')
                changed = True
            else:
                out.append(line)
        if changed:
            open(path, 'w').writelines(out)
            print(f"Successfully patched architecture mismatch in: {path}")
