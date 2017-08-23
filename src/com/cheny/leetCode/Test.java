package com.cheny.leetCode;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class Test {
	public static void main(String[] args){
		Solution solution = new Solution();
		Integer[] nums1 = new Integer[]{0,null,0,null,0,null,0};
		Integer[] nums2 = new Integer[]{5,4,7,3,1,6,2};
		int[] prenums = new int[]{1,2,3,4,5,6,7,8,9,10};
		int[] innums = new int[]{4,3,5,2,6,1,9,8,7,10};
		int[] postnums = new int[]{4,5,3,6,2,9,8,10,7,1};
		int[] nums = new int[]{1,1,8};
		String[] timePoints = new String[]{"bar","foo", "the"};
		List<String> dict = new LinkedList<>();
		dict.add("cat");
		dict.add("bat");
		dict.add("rat");
		
		char[][] board = new char[][]{"..9748...".toCharArray(), "7........".toCharArray(), ".2.1.9...".toCharArray(), "..7...24.".toCharArray(), ".64.1.59.".toCharArray(), ".98...3..".toCharArray(), "...8.3.2.".toCharArray(), "........6".toCharArray(), "...2759..".toCharArray()};
			
		Tree t1 = new Tree(nums1);
		Tree t2 = new Tree(nums2);
		int[][] num = new int[][]{{-4,-5}};
		Tree tree = new Tree(nums1);
		TreeNode root = tree.getRoot();		
		List<List<String>> res = solution.solveNQueens(3);
		
//		for(int i = 0; i < 9; i++) {
//			for(int j = 0; j < 9; j++) {
//				System.out.print(board[i][j] + " ");
//			}
//			System.out.println("");
//		}
		
		System.out.println(res);
	}
}
