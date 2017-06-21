package com.cheny.leetCode;

public class UnionFind {
	private int[] parent, rank;
	private int cnt;
	
	public UnionFind(int n){
		parent = new int[n];
		rank = new int[n];
		for(int i = 0; i < n; i++){
			parent[i] = i;
			rank[i] = 1;
		}
		cnt = n;
	}
	
	public int find(int i){
		while(i != parent[i]){
			i = parent[i];
		}
		return i;
	}
	
	public void union(int p, int q){
		int rootP = find(p);
		int rootQ = find(q);
		if(rootP == rootQ) return;
		if(rank[rootP] < rank[rootQ]){
			parent[rootP] = rootQ;
		}
		else{
			parent[rootQ] = rootP;
			if(rank[rootP] == rank[rootQ])
				rank[rootP]++;
		}
		cnt--;
	}
	
	public int count(){
		return cnt;
	}
}
