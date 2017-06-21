package com.cheny.leetCode;

import java.util.Random;

public class ShuffleAnArray {
	int[] nums;
	Random rnd;
	
	public ShuffleAnArray(int[] nums) {
        this.nums = nums.clone();
        rnd = new Random();
    }
	
	/** Resets the array to its original configuration and return it. */
    public int[] reset() {
        return nums;
    }
    
    /** Returns a random shuffling of the array. */
    public int[] shuffle() {
    	int n = nums.length;
    	int[] res = nums.clone();
        for (int i = 0; i < n; i++) {
            int r = i + rnd.nextInt(n-i);     // between i and n-1
            int temp = res[i];
            res[i] = res[r];
            res[r] = temp;
        }
        return res;
    }
}
