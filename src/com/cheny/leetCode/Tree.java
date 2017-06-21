package com.cheny.leetCode;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.Queue;

public class Tree implements Iterable<Integer> {
	private TreeNode root;

	/**
	 * Given an array, return the root of the tree.
	 * 
	 * @param vals
	 *            array of tree nodes
	 * 
	 * @return the root of the tree
	 */
	public Tree(Integer[] vals) {
		if (vals.length == 0)
			root = null;
		root = new TreeNode(vals[0]);
		Queue<TreeNode> list = new LinkedList<TreeNode>();
		list.add(root);
		for (int i = 1; i < vals.length; i += 2) {
			TreeNode p = list.poll();
			if (vals[i] != null) {
				TreeNode left = new TreeNode(vals[i]);
				p.left = left;
				list.add(p.left);
			}
			if (i + 1 < vals.length) {
				if (vals[i + 1] != null) {
					TreeNode right = new TreeNode(vals[i + 1]);
					p.right = right;
					list.add(p.right);
				}
			} else
				break;

		}
	}

	public TreeNode getRoot() {
		return root;
	}

	@Override
	public String toString() {
		return tree2str(root);

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

	@Override
	public Iterator<Integer> iterator() {
		return new TreeNodeIterator(root);
	}

	private class TreeNodeIterator implements Iterator<Integer> {
		private TreeNode cur;
		private Queue<TreeNode> q;

		public TreeNodeIterator(TreeNode cur) {
			this.cur = cur;
			q = new LinkedList<TreeNode>();
			q.add(cur);
		}

		@Override
		public boolean hasNext() {
			return !q.isEmpty();
		}

		@Override
		public Integer next() {
			if (hasNext()) {
				cur = q.poll();
				if (cur.left != null)
					q.add(cur.left);
				if (cur.right != null)
					q.add(cur.right);
				return Integer.valueOf(cur.val);
			} else
				return null;
		}

	}
}
