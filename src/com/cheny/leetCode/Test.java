package com.cheny.leetCode;

import java.util.ArrayList;
import java.util.List;

public class Test {
	public static void main(String[] args){
		Solution solution = new Solution();
		Integer[] nums1 = new Integer[]{5,2,13,1,4,6,15};
		Integer[] nums2 = new Integer[]{5,4,7,3,1,6,2};
		int[] nums = new int[]{10,1,2,7,6,1,5};
		Tree t1 = new Tree(nums1);
		Tree t2 = new Tree(nums2);
		int[][] num = new int[][]{{1,1}};
		Tree tree = new Tree(nums1);
		TreeNode root = tree.getRoot();
		List<List<Integer>> res = solution.combinationSum2(nums, 8);
//		solution.rotate(num);
		System.out.println(res);
	}
}
