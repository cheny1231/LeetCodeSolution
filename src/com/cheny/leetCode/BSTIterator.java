package com.cheny.leetCode;

import java.util.Stack;

public class BSTIterator {
	private Stack<TreeNode> stack = new Stack<>();
	
	public BSTIterator(TreeNode root) {
        pushAll(root);
    }
	
	private void pushAll(TreeNode root) {
		for(; root != null; stack.push(root), root = root.left);
	}

    /** @return whether we have a next smallest number */
    public boolean hasNext() {
        return !stack.isEmpty();
    }

    /** @return the next smallest number */
    public int next() {
        TreeNode tmp = stack.pop();
        pushAll(tmp.right);
        return tmp.val;
    }
}
