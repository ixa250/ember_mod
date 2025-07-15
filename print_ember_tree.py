import re

class TreeNode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.is_leaf = False

def print_tree(node, depth=0, prefix=""):
    indent = "  " * depth
    if node.is_leaf:
        print(f"{indent}{prefix} LEAF: {node.value:.5f}")
    else:
        print(f"{indent}{prefix} Feature[{node.feature}] <= {node.threshold}")
        print_tree(node.left, depth + 1, "L")
        print_tree(node.right, depth + 1, "R")

def parse_single_tree(text):
    print("[DEBUG] Đang phân tích một cây...")
    lines = text.strip().split('\n')
    print(f"[DEBUG] Số dòng: {len(lines)}")
    print(f"[DEBUG] Dòng đầu tiên: {lines[0]}")

    nodes = {}

    for line in lines:
        line = line.strip()
        if line.startswith("num_leaves") or not line:
            continue
        if ':' not in line:
            print(f"[DEBUG] Bỏ qua dòng không hợp lệ: {line}")
            continue

        id_part, rest = line.split(":", 1)
        try:
            node_id = int(id_part.strip())
        except ValueError:
            print(f"[DEBUG] Bỏ qua dòng không hợp lệ (ID): {line}")
            continue

        if "leaf=" in rest:
            val = float(rest.split("leaf=")[1].split()[0])
            nodes[node_id] = TreeNode(node_id)
            nodes[node_id].value = val
            nodes[node_id].is_leaf = True
        else:
            # Ex: "feature 12 <= 0.0500631 yes=1,no=2,missing=1"
            feature_part = rest.split("yes=")[0].strip()
            yes_part = re.search(r"yes=(\d+)", rest)
            no_part = re.search(r"no=(\d+)", rest)
            missing_part = re.search(r"missing=(\d+)", rest)

            try:
                feature_index = int(re.search(r"feature (\d+)", feature_part).group(1))
                threshold = float(feature_part.split("<=")[1])
            except:
                print(f"[DEBUG] Bỏ qua dòng không parse được ngưỡng: {line}")
                continue

            if not all([yes_part, no_part, missing_part]):
                print(f"[DEBUG] Thiếu nhánh con ở dòng: {line}")
                continue

            node = TreeNode(node_id)
            node.feature = feature_index
            node.threshold = threshold
            node.left = int(yes_part.group(1))
            node.right = int(no_part.group(1))
            nodes[node_id] = node

    # Liên kết các node
    for node in nodes.values():
        if not node.is_leaf:
            node.left = nodes.get(node.left)
            node.right = nodes.get(node.right)

    return nodes[0] if 0 in nodes else None

def parse_all_trees(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    tree_blocks = re.findall(r"Tree=\d+\n(.*?)(?=^Tree=\d+|\Z)", content, re.DOTALL | re.MULTILINE)
    print(f"[DEBUG] Đã phát hiện {len(tree_blocks)} block cây.")

    parsed_trees = []
    for idx, block in enumerate(tree_blocks):
        if not re.search(r"^\s*\d+\s*:", block, re.MULTILINE):
            print(f"[DEBUG] Bỏ qua block {idx} vì không có node hợp lệ.")
            continue

        print(f"[DEBUG] Đang phân tích cây {idx}...")
        root = parse_single_tree(block)
        if root:
            parsed_trees.append(root)
        else:
            print(f"[DEBUG] Bỏ qua block {idx} vì không có node hợp lệ.")

    return parsed_trees

if __name__ == "__main__":
    file_path = r"D:\\Do_An_Tot_nghiep\\Repos\\train_features\\ember2018\\ember_model_2018.txt"
    trees = parse_all_trees(file_path)

    for idx, root in enumerate(trees[:5]):
        print(f"\n===== CÂY SỐ {idx} =====")
        print_tree(root)
