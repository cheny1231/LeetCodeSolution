package com.cheny.leetCode;

import java.util.ArrayList;
import java.util.List;

public class Test {
	public static void main(String[] args){
		Solution solution = new Solution();
		Integer[] nums1 = new Integer[]{5,2,13,1,4,6,15};
		Integer[] nums2 = new Integer[]{5,4,7,3,1,6,2};
		int[] prenums = new int[]{1,2,3,4,5,6,7,8,9,10};
		int[] innums = new int[]{4,3,5,2,6,1,9,8,7,10};
		int[] postnums = new int[]{4,5,3,6,2,9,8,10,7,1};
		int[] nums = new int[]{1,0,0,0,1,0,0};
		
		Tree t1 = new Tree(nums1);
		Tree t2 = new Tree(nums2);
		int[][] num = new int[][]{{0,1}};
		Tree tree = new Tree(nums1);
		TreeNode root = tree.getRoot();
		boolean res = solution.canPlaceFlowers(nums, 2);
//		solution.rotate(num);
		System.out.println(res);
	}
}
