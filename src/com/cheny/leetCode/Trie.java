package com.cheny.leetCode;

public class Trie {
	private int R;
	private TrieNode root;
	
	public Trie(int R) {
		this.R = R;
		root = new TrieNode();
	}
	
	public void insert(String word) {
		insert(this.root, word);
	}
	
	private void insert(TrieNode root, String word) {
		if(word.isEmpty()) {
			root.isWord = true;
			return;
		}
		if(root.next[word.charAt(0)] == null) 
			root.next[word.charAt(0)] = new TrieNode();
		insert(root.next[word.charAt(0)], word.substring(1));
	}
	
	public String getShortestPre(String word) {
		int len = getShortestPre(this.root, word, -1);
		return len == 0 ? word : word.substring(0, len);
	}
	
	private int getShortestPre(TrieNode root, String word, int len) {
		if(root == null || word.isEmpty()) return 0;
		if(root.isWord) return len + 1;
		
		return getShortestPre(root.next[word.charAt(0)], word.substring(1), len + 1);
		
	}
	
	private class TrieNode {
		private boolean isWord;
		private TrieNode[] next = new TrieNode[R];
	}
}
