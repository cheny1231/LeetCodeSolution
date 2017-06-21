package com.cheny.leetCode;

public class TreeNode {
	int val;
	TreeNode left;
	TreeNode right;

	TreeNode(int x) {
		val = x;
	}

	public String toString() {
		return tree2str(this);

	}

	private String tree2str(TreeNode t) {
		if (t == null)
			return "";
		StringBuilder res = new StringBuilder();
		res.append(t.val);
		if (t.left == null && t.right == null)
			return res.toString();
		if (t.right == null)
			return res.append("(" + tree2str(t.left) + ")").toString();
		return res.append("(" + tree2str(t.left) + ")(" + tree2str(t.right) + ")").toString();
	}
	
}
