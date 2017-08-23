package com.cheny.leetCode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import java.util.stream.Collectors;

public class Solution {

	private int sumOfConvertBST = 0;

	public int numTrees(int n) {
		int[] dp = new int[n + 1];
		dp[0] = 1;
		dp[1] = 1;
		for (int i = 2; i <= n; i++) {
			for (int j = 1; j <= i; j++) {
				dp[i] += dp[j - 1] * dp[i - j];
			}
		}
		return dp[n];
	}

	public double[] calcEquation(String[][] equations, double[] values, String[][] queries) {
		Map<String, ArrayList<String>> pairs = new HashMap<>();
		Map<String, ArrayList<Double>> valuepairs = new HashMap<>();
		for (int i = 0; i < equations.length; i++) {
			String[] equation = equations[i];
			if (!pairs.containsKey(equation[0])) {
				pairs.put(equation[0], new ArrayList<>());
				valuepairs.put(equation[0], new ArrayList<>());
			}
			if (!pairs.containsKey(equation[1])) {
				pairs.put(equation[1], new ArrayList<>());
				valuepairs.put(equation[1], new ArrayList<>());
			}
			pairs.get(equation[0]).add(equation[1]);
			valuepairs.get(equation[0]).add(values[i]);
			pairs.get(equation[1]).add(equation[0]);
			valuepairs.get(equation[1]).add(1 / values[i]);
		}

		double[] res = new double[queries.length];
		for (int i = 0; i < queries.length; i++) {
			String[] query = queries[i];
			res[i] = calcEquationHelper(query[0], query[1], valuepairs, pairs, new HashSet<String>(), 1.0);
			if (res[i] == 0.0)
				res[i] = -1.0;
		}

		return res;
	}

	public double calcEquationHelper(String start, String end, Map<String, ArrayList<Double>> valuepairs,
			Map<String, ArrayList<String>> pairs, HashSet<String> set, double value) {
		if (set.contains(start))
			return 0.0;
		if (!pairs.containsKey(start))
			return 0.0;
		if (start.equals(end))
			return value;
		set.add(start);

		ArrayList<String> str = pairs.get(start);
		ArrayList<Double> val = valuepairs.get(start);

		double tmp = 0.0;

		for (int i = 0; i < str.size(); i++) {
			tmp = calcEquationHelper(str.get(i), end, valuepairs, pairs, set, value * val.get(i));
			if (tmp != 0.0)
				break;
		}

		return tmp;
	}

	public int eraseOverlapIntervals(Interval[] intervals) {
		if (intervals.length == 0)
			return 0;

		Arrays.sort(intervals, (Interval arg0, Interval arg1) -> arg0.end - arg1.end);
		int res = 0;
		int end = intervals[0].end;
		for (int i = 1; i < intervals.length; i++) {
			if (intervals[i].start < end)
				res++;
			else
				end = intervals[i].end;
		}

		return res;
	}

	/**
	 * Calculate the sum of two integers a and b, but you are not allowed to use
	 * the operator + and -.
	 * 
	 * @param two
	 *            integers
	 * 
	 * @return the sum
	 */
	public int getSum(int a, int b) {
		return b == 0 ? a : getSum(a ^ b, (a & b) << 1);
	}

	/**
	 * Given a N*N matrix M representing the friend relationship between
	 * students in the class. If M[i][j] = 1, then the ith and jth students are
	 * direct friends with each other, otherwise not. And you have to output the
	 * total number of friend circles among all the students.
	 * 
	 * @param friend
	 *            relationship
	 * 
	 * @return number of friend circles
	 */
	public int findCircleNum(int[][] M) {
		int n = M.length;
		int res = n;
		int[] unionFind = new int[n];
		int[] rank = new int[n];
		for (int i = 0; i < n; i++) {
			unionFind[i] = i;
			rank[i] = 1;
		}
		for (int i = 0; i < n; i++) {
			for (int j = i + 1; j < n; j++) {
				if (M[i][j] == 1) {
					int rootj = unionFind[j];
					while (rootj != unionFind[rootj]) {
						rootj = unionFind[rootj];
					}
					int rooti = unionFind[i];
					while (rooti != unionFind[rooti]) {
						rooti = unionFind[rooti];
					}
					if (rooti == rootj)
						continue;
					if (rank[rooti] < rank[rootj])
						unionFind[rooti] = rootj;
					else {
						unionFind[rootj] = rooti;
						if (rank[rooti] == rank[rootj])
							rank[rooti]++;
					}
					res--;
				}
			}
		}

		return res;
	}

	/**
	 * Given a binary tree, return the tilt of the whole tree.
	 * 
	 * @param root
	 *            the root of the tree
	 * 
	 * @return the tilt of the whole tree
	 */
	public int findTilt(TreeNode root) {
		if (root == null)
			return 0;
		int tilt = findTiltHelper(root, 0);
		return tilt;
	}

	private int findTiltHelper(TreeNode root, int tilt) {
		if (root.left != null)
			tilt = findTiltHelper(root.left, tilt);
		if (root.right != null)
			tilt = findTiltHelper(root.right, tilt);
		if (root.left == null && root.right == null) {
			return tilt;
		}
		if (root.left == null && root.right != null) {
			root.val += root.right.val;
			return tilt + Math.abs(root.right.val);
		}
		if (root.left != null && root.right == null) {
			root.val += root.left.val;
			return tilt + Math.abs(root.left.val);
		}
		root.val += root.right.val + root.left.val;
		return tilt + Math.abs(root.right.val - root.left.val);
	}

	public int arrayPairSum(int[] nums) {
		Arrays.sort(nums);
		int res = 0;
		for (int i = 0; i < nums.length; i = i + 2) {
			res += nums[i];
		}
		return res;
	}

	/**
	 * Given two strings representing two complex numbers. You need to return a
	 * string representing their multiplication.
	 */
	public String complexNumberMultiply(String a, String b) {
		String[] numA = a.split("\\+|i");
		String[] numB = b.split("\\+|i");
		int a1 = Integer.parseInt(numA[0]);
		int a2 = Integer.parseInt(numA[1]);
		int b1 = Integer.parseInt(numB[0]);
		int b2 = Integer.parseInt(numB[1]);
		int first = a1 * b1 - a2 * b2;
		int second = a1 * b2 + a2 * b1;
		String s = first + "+" + second + "i";
		return s;
	}

	/**
	 * Given an integer array with even length, where different numbers in this
	 * array represent different kinds of candies. Each number means one candy
	 * of the corresponding kind. You need to distribute these candies equally
	 * in number to brother and sister. Return the maximum number of kinds of
	 * candies the sister could gain.
	 */
	public int distributeCandies(int[] candies) {
		Set<Integer> s = new HashSet<>();
		for (int i : candies)
			s.add(i);
		return s.size() < (candies.length / 2) ? s.size() : (candies.length / 2);
	}

	/**
	 * The reshaped matrix need to be filled with all the elements of the
	 * original matrix in the same row-traversing order as they were.
	 */
	public int[][] matrixReshape(int[][] nums, int r, int c) {
		int oldR = nums.length;
		int oldC = nums[0].length;
		if (oldR * oldC != r * c)
			return nums;
		int[][] res = new int[r][c];
		for (int i = 0; i < oldR; i++) {
			for (int j = 0; j < oldC; j++) {
				int tmpC = (i * oldC + j) % c;
				int tmpR = (i * oldC + j) / c;
				res[tmpR][tmpC] = nums[i][j];
			}
		}
		return res;
	}

	/**
	 * Given a string, you need to reverse the order of characters in each word
	 * within a sentence while still preserving whitespace and initial word
	 * order.
	 */
	public String reverseWords(String s) {
		String[] words = s.split(" ");
		StringBuffer res = new StringBuffer();
		for (String str : words) {
			for (int i = str.length() - 1; i >= 0; i--)
				res.append(str.charAt(i));
			res.append(" ");
		}
		res.deleteCharAt(res.length() - 1);
		return res.toString();
	}

	/**
	 * You need to merge them into a new binary tree. The merge rule is that if
	 * two nodes overlap, then sum node values up as the new value of the merged
	 * node. Otherwise, the NOT null node will be used as the node of new tree.
	 * 
	 * @param t1
	 *            tree to merge
	 * @param t2
	 *            tree to merge
	 * 
	 * @return merged tree
	 */
	public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {
		if (t1 == null)
			return t2;
		if (t2 == null)
			return t1;
		TreeNode res = new TreeNode(t1.val + t2.val);
		res.left = mergeTrees(t1.left, t2.left);
		res.right = mergeTrees(t1.right, t2.right);
		return res;
	}

	/**
	 * Given a list of positive integers, the adjacent integers will perform the
	 * float division. For example, [2,3,4] -> 2 / 3 / 4. You should find out
	 * how to add parenthesis to get the maximum result
	 * 
	 * @param nums
	 *            positive integers
	 * 
	 * @return corresponding expression in string format
	 */
	public String optimalDivision(int[] nums) {
		if (nums.length == 0)
			return "";
		StringBuilder res = new StringBuilder();
		res.append(nums[0]);
		if (nums.length == 1)
			return res.toString();
		if (nums.length == 2)
			return res.append("/" + nums[1]).toString();
		res.append("/(" + nums[1]);
		for (int i = 2; i < nums.length; i++) {
			res.append("/" + nums[i]);
		}
		return res.append(")").toString();
	}

	/**
	 * Given a list of directory info including directory path, and all the
	 * files with contents in this directory, you need to find out all the
	 * groups of duplicate files in the file system in terms of their paths.
	 * 
	 * @param paths
	 *            array of Strings with the following format:
	 *            "root/d1/d2/.../dm f1.txt(f1_content) f2.txt(f2_content) ... fn.txt(fn_content)"
	 * 
	 * @return a list of group of duplicate file paths
	 * 
	 */

	public List<List<String>> findDuplicate(String[] paths) {
		if (paths.length == 0)
			return null;
		List<List<String>> res = new ArrayList<>();
		Map<String, List<String>> m = new HashMap<>();
		for (String str : paths) {
			String[] files = str.split(" ");
			String dir = files[0];
			for (int i = 1; i < files.length; i++) {
				String[] strs = files[i].split("[(]");
				List<String> list = m.get(strs[1]);
				if (list != null) {
					list.add(dir + "/" + strs[0]);
					m.put(strs[1], list);
				} else {
					list = new ArrayList<>();
					list.add(dir + "/" + strs[0]);
					m.put(strs[1], list);
				}
			}

		}
		for (Map.Entry<String, List<String>> entry : m.entrySet()) {
			if (entry.getValue().size() > 1)
				res.add(entry.getValue());
		}
		return res;
	}

	/**
	 * 
	 * */
	public TreeNode convertBST(TreeNode root) {
		convertBSTHelper(root);
		return root;

	}

	private void convertBSTHelper(TreeNode root) {
		if (root == null)
			return;

		convertBSTHelper(root.right);

		root.val += sumOfConvertBST;
		sumOfConvertBST = root.val;

		convertBSTHelper(root.left);
	}

	/**
	 * find out their common interest with the least list index sum
	 * 
	 * @param list1
	 *            first list of restaurant
	 * @param list2
	 *            second list of restaurant
	 * 
	 * @return the common ones with the least list index sum
	 */

	public String[] findRestaurant(String[] list1, String[] list2) {
		List<String> res = new LinkedList<>();
		int minSum = Integer.MAX_VALUE;
		Map<String, Integer> m = new HashMap<>();
		for (int i = 0; i < list1.length; i++)
			m.put(list1[i], i);
		for (int i = 0; i < list2.length; i++) {
			Integer j = m.get(list2[i]);
			if (j != null && j + i <= minSum) {
				if (i + j < minSum) {
					res = new LinkedList<>();
					minSum = j + i;
				}
				res.add(list2[i]);
			}
		}
		return res.toArray(new String[res.size()]);
	}

	public int arrayNesting(int[] nums) {
		int res = 0;
		for (int i = 0; i < nums.length; i++) {
			int temp = 0;
			for (int k = i; nums[k] >= 0; temp++) {
				int numsk = nums[k];
				nums[k] = -1;
				k = numsk;
			}
			if (temp > res)
				res = temp;
		}
		return res;
	}

	public int subarraySum(int[] nums, int k) {
		int res = 0;
		int sum = 0;
		Map<Integer, Integer> sums = new HashMap<>();
		sums.put(0, 1);
		for (int i = 0; i < nums.length; i++) {
			sum += nums[i];
			if (sums.containsKey(sum - k))
				res += sums.get(sum - k);
			sums.put(sum, sums.getOrDefault(sum, 0) + 1);
		}
		return res;
	}

	public int maxProfit(int[] prices) {
		int res = 0, cur = 0;
		for (int i = 1; i < prices.length; i++) {
			cur += prices[i] - prices[i - 1];
			if (cur < 0)
				cur = 0;
			if (cur > res)
				res = cur;
		}
		return res;
	}

	public int uniquePaths(int m, int n) {
		if (m > n)
			return uniquePaths(n, m);
		int[] paths = new int[m];
		for (int i = 0; i < m; i++) {
			paths[i] = 1;
		}

		for (int i = 1; i < n; i++) {
			for (int j = 1; j < m; j++) {
				paths[j] += paths[j - 1];
			}
		}
		return paths[m - 1];
	}

	public int findMin(int[] nums) {
		int low = 0, high = nums.length - 1;
		while (low < high) {
			if (nums[low] < nums[high])
				return nums[low];
			int mid = (low + high) / 2;
			if (nums[low] <= nums[mid])
				low = mid + 1;
			else
				high = mid;
		}

		return nums[low];
	}

	public int searchInsert(int[] nums, int target) {
		int low = 0, high = nums.length - 1;
		while (low <= high) {
			int mid = (low + high) / 2;
			if (target < nums[mid])
				high = mid - 1;
			else if (target > nums[mid])
				low = mid + 1;
			else
				return mid;
		}
		return low;
	}

	public List<List<Integer>> subsets(int[] nums) {
		List<List<Integer>> res = new ArrayList<>();
		List<Integer> cur = new ArrayList<>();
		subsetsHelper(nums, res, cur, 0);
		return res;
	}

	private void subsetsHelper(int[] nums, List<List<Integer>> res, List<Integer> cur, int i) {
		if (i == nums.length) {
			res.add(new ArrayList<Integer>(cur));
			return;
		}
		cur.add(nums[i]);
		subsetsHelper(nums, res, cur, i + 1);
		cur.remove(cur.size() - 1);
		subsetsHelper(nums, res, cur, i + 1);
	}

	public int maxSubArray(int[] nums) {
		int res = Integer.MIN_VALUE;
		int cur = 0;
		for (int i = 0; i < nums.length; i++) {
			if (cur < 0)
				cur = 0;
			cur += nums[i];
			if (cur > res)
				res = cur;
		}
		return res;
	}

	public int[][] generateMatrix(int n) {
		int[][] res = new int[n][n];

		if (n == 0)
			return res;

		int rowStart = 0, rowEnd = n - 1;
		int colStart = 0, colEnd = n - 1;
		int num = 1;

		while (rowStart <= rowEnd && colStart <= colEnd) {
			for (int i = colStart; i <= colEnd; i++) {
				res[rowStart][i] = num++;
			}
			rowStart++;

			for (int i = rowStart; i <= rowEnd; i++) {
				res[i][colEnd] = num++;
			}
			colEnd--;

			if (rowStart <= rowEnd) {
				for (int i = colEnd; i >= colStart; i--) {
					res[rowEnd][i] = num++;
				}
			}
			rowEnd--;

			if (colStart <= colEnd) {
				for (int i = rowEnd; i >= rowStart; i--) {
					res[i][colStart] = num++;
				}
			}
			colStart++;
		}

		return res;
	}

	public int removeElement(int[] nums, int val) {
		int end = nums.length - 1;
		for (int i = 0; i < nums.length; i++) {
			if (i < end && nums[i] == val) {
				nums[i] = nums[end--];
				i--;
			}
		}
		return end + 1;
	}

	public int[] plusOne(int[] digits) {
		for (int i = digits.length - 1; i >= 0; i--) {
			digits[i]++;
			if (digits[i] < 10) {
				return digits;
			} else {
				digits[i] = 0;
			}
		}

		int[] res = new int[digits.length + 1];
		res[0] = 1;
		return res;
	}

	public void rotate(int[][] matrix) {
		int n = matrix.length;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n / 2; j++) {
				int tmp = matrix[j][i];
				matrix[j][i] = matrix[n - j - 1][i];
				matrix[n - j - 1][i] = tmp;
			}
		}
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < i; j++) {
				int tmp = matrix[j][i];
				matrix[j][i] = matrix[i][j];
				matrix[i][j] = tmp;
			}
		}
	}

	public int minPathSum(int[][] grid) {
		int row = grid.length;
		int col = grid[0].length;

		/** O(n^2) space dp */
		// int[][] dp = new int[row][col];
		// dp[0][0] = grid[0][0];
		// for (int i = 1; i < col; i++) {
		// dp[0][i] = dp[0][i - 1] + grid[0][i];
		// }
		// for (int i = 1; i < row; i++) {
		// dp[i][0] = dp[i - 1][0] + grid[i][0];
		// }
		// for (int i = 1; i < row; i++) {
		// for (int j = 1; j < col; j++) {
		// dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
		// }
		// }
		// return dp[row - 1][col - 1];

		/** O(1) space dp */
		// for (int i = 1; i < col; i++) {
		// grid[0][i] = grid[0][i - 1] + grid[0][i];
		// }
		// for (int i = 1; i < row; i++) {
		// grid[i][0] = grid[i - 1][0] + grid[i][0];
		// }
		// for (int i = 1; i < row; i++) {
		// for (int j = 1; j < col; j++) {
		// grid[i][j] = Math.min(grid[i - 1][j], grid[i][j - 1]) + grid[i][j];
		// }
		// }
		// return grid[row - 1][col - 1];

		/** O(n) space dp */
		int[] dp = new int[col];
		dp[0] = grid[0][0];
		for (int i = 1; i < col; i++) {
			dp[i] = dp[i - 1] + grid[0][i];
		}
		for (int i = 1; i < row; i++) {
			dp[0] += grid[i][0];
			for (int j = 1; j < col; j++) {
				dp[j] = Math.min(dp[j], dp[j - 1]) + grid[i][j];
			}
		}
		return dp[col - 1];
	}

	public List<List<Integer>> generate(int numRows) {
		List<List<Integer>> res = new ArrayList<>(numRows);

		if (numRows == 0)
			return res;

		List<Integer> first = new ArrayList<>(1);
		first.add(1);
		res.add(first);

		for (int curSize = 1; curSize < numRows; curSize++) {
			List<Integer> cur = new ArrayList<>();
			cur.add(1);
			for (int i = 1; i < curSize; i++) {
				cur.add(res.get(curSize - 1).get(i - 1) + res.get(curSize - 1).get(i));
			}
			cur.add(1);
			res.add(cur);
		}

		return res;
	}

	public List<List<Integer>> combinationSum(int[] candidates, int target) {
		List<List<Integer>> res = new LinkedList<>();

		combinationSumHelper(res, new LinkedList<>(), 0, candidates, target);

		return res;
	}

	private void combinationSumHelper(List<List<Integer>> res, List<Integer> cur, int loc, int[] candidates,
			int target) {
		if (target == 0) {
			res.add(new LinkedList<Integer>(cur));
			return;
		}

		for (; loc < candidates.length; loc++) {
			if (target >= candidates[loc]) {
				cur.add(candidates[loc]);

				combinationSumHelper(res, cur, loc, candidates, target - candidates[loc]);
				cur.remove(cur.size() - 1);
			}
		}
	}

	public void sortColors(int[] nums) {
		int second = nums.length - 1;
		int zero = 0;

		for (int i = 0; i <= second; i++) {
			while (nums[i] == 2 && i < second) {
				int tmp = nums[second];
				nums[second--] = nums[i];
				nums[i] = tmp;
			}
			while (nums[i] == 0 && i > zero) {
				int tmp = nums[zero];
				nums[zero++] = nums[i];
				nums[i] = tmp;
			}
		}
	}

	public int findPeakElement(int[] nums) {
		int low = 0;
		int high = nums.length - 1;
		while (low < high) {
			int midl = (low + high) / 2;
			int midr = midl + 1;
			if (nums[midl] < nums[midr])
				low = midr;
			else
				high = midl;
		}
		return low;
	}

	public void gameOfLife(int[][] board) {
		if (board == null || board.length == 0)
			return;

		int row = board.length;
		int col = board[0].length;

		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				int lives = liveNeighbors(board, row, col, i, j);
				if (board[i][j] == 1 && (lives == 3 || lives == 2))
					board[i][j] = 3;
				if (board[i][j] == 0 && lives == 3)
					board[i][j] = 2;

			}
		}

		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				board[i][j] >>= 1;
			}
		}
	}

	private int liveNeighbors(int[][] board, int row, int col, int i, int j) {
		int lives = 0;

		for (int x = Math.max(0, i - 1); x < i + 2 && x < row; x++) {
			for (int y = Math.max(0, j - 1); y < j + 2 && y < col; y++) {
				lives += board[x][y] & 1;
			}
		}

		lives -= board[i][j] & 1;

		return lives;
	}

	public int maxArea(int[] height) {
		int left = 0, right = height.length - 1;
		int res = 0;

		while (left < right) {
			res = Math.max(res, Math.min(height[left], height[right]) * (right - left));
			if (height[left] < height[right])
				left++;
			else
				right--;
		}

		return res;
	}

	public int trap(int[] height) {
		int left = 0, right = height.length - 1;
		int leftMax = 0, rightMax = 0;
		int res = 0;

		while (left < right) {
			if (height[left] < height[right]) {
				if (height[left] > leftMax)
					leftMax = height[left];
				else
					res += leftMax - height[left];
				left++;
			} else {
				if (height[right] > rightMax)
					rightMax = height[right];
				else
					res += rightMax - height[right];
				right--;
			}
		}

		return res;
	}

	public List<Integer> getRow(int rowIndex) {
		List<Integer> res = new ArrayList<>(rowIndex + 1);

		res.add(1);

		for (int i = 1; i <= rowIndex; i++) {
			res.add(0);
		}

		for (int i = 0; i <= rowIndex; i++) {
			for (int j = i; j > 0; j--) {
				res.set(j, res.get(j - 1) + res.get(j));
			}
		}

		return res;
	}

	public int triangleNumber(int[] nums) {
		Arrays.sort(nums);
		int n = nums.length;
		int j = 1, k = n - 1;
		int res = 0;

		for (int i = n - 1; i > 1; i--) {
			j = 0;
			k = i - 1;
			while (j < k) {
				if (nums[j] + nums[k] > nums[i]) {
					res += k - j;
					k--;
				} else
					j++;
			}
		}

		return res;
	}

	public int longestConsecutive(int[] nums) {
		if (nums.length == 0)
			return 0;

		Arrays.sort(nums);

		int res = 0;
		int cur = 1;

		for (int i = 1; i < nums.length; i++) {
			if (nums[i] - nums[i - 1] == 1) {
				cur++;
			} else if (nums[i] == nums[i - 1]) {
				continue;
			} else {
				res = res > cur ? res : cur;
				cur = 1;
			}
		}

		res = res > cur ? res : cur;

		return res;
	}

	public int removeDuplicates(int[] nums) {
		if (nums.length < 2)
			return nums.length;

		int res = 1;
		int l = 1;

		for (; l < nums.length; l++) {
			if (nums[l - 1] != nums[l]) {
				nums[res++] = nums[l];
			} else
				continue;
		}

		return res;
	}

	public int removeDuplicatesII(int[] nums) {
		if (nums.length < 2)
			return nums.length;

		int res = 1;
		int cur = 1;
		int l = 1;

		for (; l < nums.length; l++) {
			if (nums[l - 1] != nums[l]) {
				nums[res++] = nums[l];
				cur = 1;
			} else if (cur < 2) {
				nums[res++] = nums[l];
				cur++;
			} else {
				cur++;
				continue;
			}
		}

		return res;
	}

	public List<List<Integer>> subsetsWithDup(int[] nums) {
		List<List<Integer>> res = new ArrayList<>();

		List<Integer> n = new ArrayList<>();
		res.add(n);

		Arrays.sort(nums);

		for (int i = 0; i < nums.length;) {
			int cnt = 0;
			while (i + cnt < nums.length && nums[i + cnt] == nums[i])
				cnt++;
			int curSize = res.size();
			for (int j = 0; j < curSize; j++) {
				List<Integer> cur = new ArrayList<>(res.get(j));
				for (int k = 0; k < cnt; k++) {
					cur.add(nums[i]);
					res.add(new ArrayList<>(cur));
				}
			}
			i += cnt;
		}

		return res;
	}

	public boolean searchMatrix(int[][] matrix, int target) {
		int row = matrix.length;
		if (row == 0)
			return false;
		int col = matrix[0].length;
		if (col == 0)
			return false;
		int l = 0, r = row * col - 1;

		while (l < r) {
			int mid = (l + r - 1) >> 1;
			if (matrix[mid / col][mid % col] < target)
				l = mid + 1;
			else
				r = mid;
		}

		return matrix[r / col][r % col] == target;
	}

	public int[] twoSum(int[] nums, int target) {
		Map<Integer, Integer> m = new HashMap<>();

		for (int i = 0; i < nums.length; i++) {
			if (m.containsKey(target - nums[i])) {
				return new int[] { m.get(target - nums[i]), i };
			}
			m.put(nums[i], i);
		}

		return null;
	}

	public int minimumTotal(List<List<Integer>> triangle) {
		// return minimumTotalHelper(Integer.MAX_VALUE, triangle.get(0).get(0),
		// 1, 0, triangle);

		/** DP Solution */
		int n = triangle.size();
		int[] dp = new int[n];
		for (int i = 0; i < n; i++) {
			dp[i] = triangle.get(n - 1).get(i);
		}
		for (int layer = n - 2; layer >= 0; layer--) {
			for (int i = 0; i < layer + 1; i++) {
				dp[i] = Math.min(dp[i], dp[i + 1]) + triangle.get(layer).get(i);
			}
		}
		return dp[0];
	}

	// private int minimumTotalHelper(int min, int res, int cur, int pos,
	// List<List<Integer>> triangle) {
	// if (cur == triangle.size()) {
	// if (res < min)
	// return res;
	// else
	// return min;
	// }
	//
	// List<Integer> curList = triangle.get(cur);
	// res += curList.get(pos);
	// min = minimumTotalHelper(min, res, cur + 1, pos, triangle);
	//
	// if (pos < curList.size() - 1) {
	// res = res - curList.get(pos) + curList.get(pos + 1);
	// min = minimumTotalHelper(min, res, cur + 1, pos + 1, triangle);
	// }
	//
	// return min;
	// }

	public List<List<Integer>> combinationSum2(int[] candidates, int target) {
		List<List<Integer>> res = new ArrayList<>();
		List<Integer> cur = new ArrayList<Integer>();
		Arrays.sort(candidates);
		combinationSum2Helper(candidates, target, 0, cur, res, 0);
		return res;
	}

	private void combinationSum2Helper(int[] candidates, int target, int sum, List<Integer> cur,
			List<List<Integer>> res, int pos) {
		if (sum == target) {
			res.add(new ArrayList<Integer>(cur));
			return;
		}
		for (; pos < candidates.length; pos++) {
			if (sum + candidates[pos] <= target) {
				cur.add(candidates[pos]);
				combinationSum2Helper(candidates, target, sum + candidates[pos], cur, res, pos + 1);
				cur.remove(cur.size() - 1);
				while (pos + 1 < candidates.length && candidates[pos + 1] == candidates[pos])
					pos++;
			} else
				return;
		}
	}

	public int maxDistance(List<List<Integer>> arrays) {
		int max = arrays.get(0).get(arrays.get(0).size() - 1);
		int min = arrays.get(0).get(0);
		int result = Integer.MIN_VALUE;

		for (int i = 1; i < arrays.size(); i++) {
			List<Integer> cur = arrays.get(i);
			result = Math.max(result, max - cur.get(0));
			result = Math.max(result, cur.get(cur.size() - 1) - min);
			max = Math.max(max, cur.get(cur.size() - 1));
			min = Math.min(min, cur.get(0));
		}

		return result;
	}

	public boolean containsNearbyDuplicate(int[] nums, int k) {
		/** Hash Map */
		// Map<Integer, Integer> m = new HashMap<>();
		// for(int i = 0; i < nums.length; i++){
		// if(m.containsKey(nums[i])){
		// if(i - m.get(nums[i]) <= k) return true;
		// }
		// m.put(nums[i], i);
		// }
		//
		// return false;

		/** Hash Set */
		Set<Integer> set = new HashSet<>();
		for (int i = 0; i < nums.length; i++) {
			if (i >= k)
				set.remove(nums[i - k - 1]);
			if (!set.add(nums[i]))
				return true;
		}
		return false;
	}

	public TreeNode buildTree(int[] preorder, int[] inorder) {
		if (preorder.length == 0 || preorder.length != inorder.length)
			return null;
		return buildTreeHelper(preorder, inorder, 0, 0, preorder.length - 1);
	}

	private TreeNode buildTreeHelper(int[] preorder, int[] inorder, int prestart, int instart, int inend) {
		if (prestart >= preorder.length || instart > inend)
			return null;

		TreeNode root = new TreeNode(preorder[prestart]);

		if (instart == inend)
			return root;

		int pos = instart;
		for (; pos <= inend; pos++) {
			if (inorder[pos] == preorder[prestart])
				break;
		}

		root.left = buildTreeHelper(preorder, inorder, prestart + 1, instart, pos - 1);
		root.right = buildTreeHelper(preorder, inorder, pos - instart + prestart + 1, pos + 1, inend);

		return root;
	}

	public void merge(int[] nums1, int m, int[] nums2, int n) {
		int i = m - 1, j = n - 1, k = m + n - 1;
		while (i >= 0 && j >= 0) {
			if (nums1[i] < nums2[j])
				nums1[k--] = nums2[j--];
			else
				nums1[k--] = nums1[i--];
		}
		while (j >= 0)
			nums1[k--] = nums2[j--];
	}

	public TreeNode buildTreeII(int[] inorder, int[] postorder) {
		return buildTreeIIHelper(inorder, postorder, 0, inorder.length - 1, inorder.length - 1);
	}

	private TreeNode buildTreeIIHelper(int[] inorder, int[] postorder, int instart, int inend, int postend) {
		if (postend < 0 || instart > inend)
			return null;

		TreeNode root = new TreeNode(postorder[postend]);
		if (instart == inend)
			return root;

		int pos = instart;
		for (; pos <= inend; pos++) {
			if (inorder[pos] == postorder[postend])
				break;
		}

		root.left = buildTreeIIHelper(inorder, postorder, instart, pos - 1, postend - inend + pos - 1);
		root.right = buildTreeIIHelper(inorder, postorder, pos + 1, inend, postend - 1);

		return root;
	}

	public int uniquePathsWithObstacles(int[][] obstacleGrid) {
		if (obstacleGrid[0][0] == 1)
			return 0;

		int n = obstacleGrid[0].length;

		int[] dp = new int[n];
		for (int i = 0; i < n; i++) {
			if (obstacleGrid[0][i] == 1)
				break;
			dp[i] = 1;
		}

		for (int layer = 1; layer < obstacleGrid.length; layer++) {
			if (obstacleGrid[layer][0] == 1)
				dp[0] = 0;

			for (int i = 1; i < n; i++) {
				if (obstacleGrid[layer][i - 1] == 0 && obstacleGrid[layer][i] == 0)
					dp[i] = dp[i] + dp[i - 1];
				else if (obstacleGrid[layer][i] == 1)
					dp[i] = 0;
			}
		}

		return dp[n - 1];
	}

	public int[] searchRange(int[] nums, int target) {
		int l = 0, r = nums.length - 1;

		while (l <= r) {
			int mid = l + (r - l) / 2;
			if (nums[mid] < target)
				l = mid + 1;
			else if (nums[mid] > target)
				r = mid - 1;
			else {
				l = mid - 1;
				r = mid + 1;
				while (l >= 0 && nums[l] == target)
					l--;
				while (r < nums.length && nums[r] == target)
					r++;
				return new int[] { l + 1, r - 1 };
			}
		}
		return new int[] { -1, -1 };
	}

	public int threeSumClosest(int[] nums, int target) {
		Arrays.sort(nums);

		int res = 0;

		if (nums.length < 4) {
			for (int i : nums)
				res += i;
			return res;
		}

		res = nums[0] + nums[1] + nums[2];
		for (int i = 0; i < nums.length - 2; i++) {
			int j = i + 1;
			int k = nums.length - 1;
			while (j < k) {
				int sum = nums[j] + nums[k] + nums[i];
				if (Math.abs(res - target) > Math.abs(sum - target))
					res = sum;
				if (sum < target)
					j++;
				else if (sum > target)
					k--;
				else
					return sum;
			}
		}
		return res;
	}

	public boolean canPlaceFlowers(int[] flowerbed, int n) {
		for (int i = 0; i < flowerbed.length; i++) {
			if ((i - 1 < 0 || flowerbed[i - 1] == 0) && flowerbed[i] == 0
					&& (i + 1 >= flowerbed.length || flowerbed[i + 1] == 0)) {
				flowerbed[i] = 1;
				n--;
			}
			if (n == 0)
				return true;
		}
		if (n <= 0)
			return true;
		return false;
	}

	public int minSubArrayLen(int s, int[] nums) {
		if (nums.length == 0)
			return 0;
		int i = 0, j = 0;
		int sum = 0;
		int min = Integer.MAX_VALUE;

		while (j < nums.length) {
			sum += nums[j++];

			while (sum >= s) {
				if (j - i < min)
					min = j - i;
				sum -= nums[i++];
			}
		}
		return sum == Integer.MAX_VALUE ? 0 : min;
	}

	public int findUnsortedSubarray(int[] nums) {
		int begin = -1, end = -2;
		int max = nums[0];
		int min = nums[nums.length - 1];

		for (int i = 1; i < nums.length; i++) {
			if (nums[i] < max)
				end = i;
			if (nums[nums.length - 1 - i] > min)
				begin = i;
			max = Math.max(max, nums[i]);
			min = Math.min(min, nums[nums.length - 1 - i]);
		}

		return end - begin + 1;
	}

	public List<Interval> merge(List<Interval> intervals) {
		if (intervals.size() == 0)
			return intervals;

		intervals.sort((i1, i2) -> Integer.compare(i1.start, i2.start));
		int start = intervals.get(0).start;
		int end = intervals.get(0).end;
		List<Interval> res = new LinkedList<Interval>();

		for (Interval i : intervals) {
			if (i.start <= end)
				end = Math.max(i.end, end);
			else {
				res.add(new Interval(start, end));
				start = i.start;
				end = i.end;
			}
		}

		res.add(new Interval(start, end));

		return res;
	}

	public boolean canJump(int[] nums) {
		int i = 0;

		for (int reach = 0; i < nums.length && i <= reach; i++) {
			reach = Math.max(reach, i + nums[i]);
		}

		return i >= nums.length;
	}

	public List<String> summaryRanges(int[] nums) {
		List<String> res = new LinkedList<>();

		if (nums.length == 0)
			return res;

		for (int i = 0; i < nums.length; i++) {
			int a = nums[i];

			while (i + 1 < nums.length && nums[i + 1] - nums[i] == 1) {
				i++;
			}

			if (a == nums[i])
				res.add(a + "");
			else
				res.add(a + "->" + nums[i]);
		}

		return res;
	}

	public int findMinDifference(List<String> timePoints) {
		boolean[] times = new boolean[1440];
		for (String s : timePoints) {
			String[] strs = s.split(":");
			int h = Integer.parseInt(strs[0]);
			int m = Integer.parseInt(strs[1]);
			if (times[h * 60 + m])
				return 0;
			else
				times[h * 60 + m] = true;
		}

		int prev = -1440, first = -1, end = -1, res = 1440;
		for (int i = 0; i < times.length; i++) {
			if (times[i]) {
				if (first < 0)
					first = i;
				end = Math.max(end, i);
				res = Math.min(res, i - prev);
				prev = i;
			}
		}

		return Math.min(res, 1440 - end + first);
	}

	public String reverseStr(String s, int k) {
		char[] c = s.toCharArray();
		int i = 0;

		while (i < s.length()) {
			int j = Math.min(i + k - 1, s.length() - 1);
			reverse(c, i, j);
			i += 2 * k;
		}

		return String.valueOf(c);
	}

	private void reverse(char[] c, int i, int j) {
		while (i < j) {
			char tmp = c[i];
			c[i++] = c[j];
			c[j--] = tmp;
		}
	}

	private void swap(int[] nums, int i, int j) {
		int tmp = nums[i];
		nums[i] = nums[j];
		nums[j] = tmp;
	}

	public void nextPermutation(int[] nums) {
		int i = nums.length - 1;

		while (i > 0 && nums[i] <= nums[i - 1])
			i--;

		if (i == 0) {
			Arrays.sort(nums);
			return;
		}

		int min = Integer.MAX_VALUE;
		int ind = i;

		for (int j = i; j < nums.length; j++) {
			if (nums[j] > nums[i - 1] && nums[j] - nums[i - 1] < min) {
				min = nums[j] - nums[i - 1];
				ind = j;
			}
		}

		swap(nums, i - 1, ind);
		Arrays.sort(nums, i, nums.length);

	}

	public List<Integer> majorityElement(int[] nums) {
		List<Integer> res = new ArrayList<>();

		if (nums.length == 0)
			return res;

		int maj = nums.length / 3;
		Map<Integer, Integer> m = new HashMap<>();

		for (int i : nums) {
			if (m.containsKey(i)) {
				int cnt = m.get(i);
				m.put(i, cnt + 1);
			} else
				m.put(i, 1);
		}

		for (Map.Entry<Integer, Integer> entry : m.entrySet()) {
			if (entry.getValue() > maj)
				res.add(entry.getKey());
		}

		return res;
	}

	public int findPairs(int[] nums, int k) {
		int res = 0;
		Arrays.sort(nums);

		for (int i = 0; i < nums.length; i++) {
			int j = i + 1;
			while (j < nums.length && nums[j] - nums[i] <= k) {
				if (nums[j] - nums[i] == k) {
					res++;
					break;
				}
				j++;
			}
			while (i + 1 < nums.length && nums[i + 1] == nums[i])
				i++;
		}

		return res;
	}

	public int thirdMax(int[] nums) {
		Integer max = null;
		Integer semax = max;
		Integer thmax = max;

		for (Integer i : nums) {
			if (i.equals(max) || i.equals(semax) || i.equals(thmax))
				continue;
			if (max == null || i > max) {
				thmax = semax;
				semax = max;
				max = i;
			} else if (semax == null || (i < max && i > semax)) {
				thmax = semax;
				semax = i;
			} else if (thmax == null || (i < semax && i > thmax)) {
				thmax = i;
			}
		}

		return thmax == null ? max : thmax;
	}

	private void reverse(int[] c, int i, int j) {
		while (i < j) {
			int tmp = c[i];
			c[i++] = c[j];
			c[j--] = tmp;
		}
	}

	public void rotate(int[] nums, int k) {
		k = k % nums.length;

		reverse(nums, 0, nums.length - 1);
		reverse(nums, 0, k - 1);
		reverse(nums, k, nums.length - 1);

	}

	public int maximumProduct(int[] nums) {
		Integer max = Integer.MIN_VALUE;
		Integer semax = Integer.MIN_VALUE;
		Integer thmax = Integer.MIN_VALUE;
		Integer min = Integer.MAX_VALUE;
		Integer semin = Integer.MAX_VALUE;

		for (Integer i : nums) {
			if (i >= max) {
				thmax = semax;
				semax = max;
				max = i;
			} else if (i >= semax) {
				thmax = semax;
				semax = i;
			} else if (i >= thmax) {
				thmax = i;
			}
			if (i <= min) {
				semin = min;
				min = i;
			} else if (i <= semin) {
				semin = i;
			}
		}

		return Integer.max(max * semax * thmax, max * min * semin);
	}

	public void setZeroes(int[][] matrix) {
		int row = matrix.length;
		int col = matrix[0].length;
		boolean[] cols = new boolean[col];
		boolean[] rows = new boolean[row];

		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				if (matrix[i][j] == 0) {
					cols[j] = true;
					rows[i] = true;
				}
			}
		}

		for (int i = 0; i < row; i++) {
			if (rows[i])
				setRowZeroes(matrix, i);
		}

		for (int i = 0; i < col; i++) {
			if (cols[i])
				setColZeroes(matrix, i);
		}
	}

	private void setRowZeroes(int[][] matrix, int row) {
		if (row < matrix.length) {
			for (int i = 0; i < matrix[0].length; i++) {
				matrix[row][i] = 0;
			}
		}
	}

	private void setColZeroes(int[][] matrix, int col) {
		if (col < matrix[0].length) {
			for (int i = 0; i < matrix.length; i++) {
				matrix[i][col] = 0;
			}
		}
	}

	public List<List<Integer>> fourSum(int[] nums, int target) {
		List<List<Integer>> res = new LinkedList<>();

		if (nums.length < 4)
			return res;

		Arrays.sort(nums);

		if (4 * nums[0] > target || 4 * nums[nums.length - 1] < target)
			return res;

		fourSumHelper(res, new LinkedList<Integer>(), 0, nums, 0, target);

		return res;
	}

	private void fourSumHelper(List<List<Integer>> res, List<Integer> cur, int i, int[] nums, int sum, int target) {
		if (sum == target && cur.size() == 4) {
			res.add(new LinkedList<Integer>(cur));
			return;
		} else if (cur.size() >= 4)
			return;

		for (; i < nums.length; i++) {
			if (nums[i] + 3 * nums[nums.length - 1] < target)
				continue;
			if (nums[i] + 3 * nums[0] > target)
				break;
			sum += nums[i];
			cur.add(nums[i]);
			fourSumHelper(res, cur, i + 1, nums, sum, target);
			sum -= nums[i];
			cur.remove(cur.size() - 1);

			while (i < nums.length - 1 && nums[i] == nums[i + 1])
				i++;
		}
	}

	public boolean exist(char[][] board, String word) {
		for (int i = 0; i < board.length; i++) {
			for (int j = 0; j < board[0].length; j++) {
				if (existHelper(board, word, 0, i, j))
					return true;
			}
		}
		return false;
	}

	private boolean existHelper(char[][] board, String word, int pos, int row, int col) {
		if (pos == word.length())
			return true;

		if (row < 0 || row == board.length || col < 0 || col == board[0].length)
			return false;

		boolean flag;

		if (board[row][col] == word.charAt(pos)) {
			char c = board[row][col];
			board[row][col] = 0;
			flag = existHelper(board, word, pos + 1, row - 1, col) || existHelper(board, word, pos + 1, row, col - 1)
					|| existHelper(board, word, pos + 1, row + 1, col)
					|| existHelper(board, word, pos + 1, row, col + 1);
			board[row][col] = c;
		} else {
			return false;
		}

		return flag;
	}

	public boolean PredictTheWinner(int[] nums) {
		return PredictTheWinnerHelper(nums, 0, nums.length - 1, new Integer[nums.length][nums.length]) >= 0;
	}

	private int PredictTheWinnerHelper(int[] nums, int s, int e, Integer[][] dp) {
		if (dp[s][e] == null) {
			dp[s][e] = s == e ? nums[s]
					: Math.max(nums[e] - PredictTheWinnerHelper(nums, s, e - 1, dp),
							nums[s] - PredictTheWinnerHelper(nums, s + 1, e, dp));
		}
		return dp[s][e];
	}

	public int maxProfitCooldown(int[] prices) {
		if (prices.length == 0)
			return 0;

		int[] s0 = new int[prices.length];
		int[] s1 = new int[prices.length];
		int[] s2 = new int[prices.length];

		s1[0] = -prices[0];

		for (int i = 1; i < prices.length; i++) {
			s0[i] = Math.max(s0[i - 1], s2[i - 1]);
			s1[i] = Math.max(s0[i - 1] - prices[i], s1[i - 1]);
			s2[i] = s1[i - 1] + prices[i];
		}

		return Math.max(s0[prices.length - 1], s2[prices.length - 1]);
	}

	public int climbStairs(int n) {
		int[] dp = new int[3];

		dp[0] = 1;
		dp[1] = 1;

		for (int i = 2; i < n + 1; i++) {
			dp[2] = dp[0] + dp[1];
			dp[0] = dp[1];
			dp[1] = dp[2];
		}

		return n > 2 ? dp[2] : dp[n];
	}

	public int rob(int[] nums) {
		int s1 = 0;
		int s2 = 0, tmp;

		for (int i = 0; i < nums.length; i++) {
			tmp = s2;
			s2 = s1 + nums[i];
			s1 = Math.max(s1, tmp);
		}

		return Math.max(s1, s2);
	}

	public int lengthOfLIS(int[] nums) {
		int[] dp = new int[nums.length];
		int len = 0;

		for (int x : nums) {
			int i = Arrays.binarySearch(dp, 0, len, x);
			if (i < 0)
				i = -i - 1;
			dp[i] = x;
			if (i == len)
				len++;
		}

		return len;
	}

	public int findMaxForm(String[] strs, int m, int n) {
		int[][] dp = new int[m + 1][n + 1];

		for (String str : strs) {
			int ones = 0, zeroes = 0;
			for (char c : str.toCharArray()) {
				if (c == '1')
					ones++;
				else
					zeroes++;
			}

			for (int i = m; i >= zeroes; i--) {
				for (int j = n; j >= ones; j--) {
					dp[i][j] = Math.max(dp[i][j], dp[i - zeroes][j - ones] + 1);
				}
			}
		}

		return dp[m][n];
	}

	public int shoppingOffers(List<Integer> price, List<List<Integer>> special, List<Integer> needs) {
		int res = Integer.MAX_VALUE;

		for (int i = 0; i < special.size(); i++) {
			List<Integer> sp = special.get(i);
			boolean validOffer = true;

			for (int j = 0; j < needs.size(); j++) {
				int tmp = needs.get(j) - sp.get(j);
				if (tmp < 0)
					validOffer = false;
				needs.set(j, tmp);
			}

			if (validOffer) {
				res = Math.min(res, shoppingOffers(price, special, needs) + sp.get(sp.size() - 1));
			}

			for (int j = 0; j < needs.size(); j++) {
				int tmp = needs.get(j) + sp.get(j);
				needs.set(j, tmp);
			}
		}

		int cost = 0;
		for (int j = 0; j < needs.size(); j++) {
			cost += needs.get(j) * price.get(j);
		}

		return Math.min(res, cost);
	}

	public int getMoneyAmount(int n) {
		int[][] dp = new int[n + 1][n + 1];
		return getMoneyAmountHelper(dp, 1, n);
	}

	private int getMoneyAmountHelper(int[][] dp, int i, int j) {
		if (i >= j)
			return 0;
		if (dp[i][j] != 0)
			return dp[i][j];

		int res = Integer.MAX_VALUE;

		for (int k = i; k <= j; k++) {
			int tmp = k + Integer.max(getMoneyAmountHelper(dp, i, k - 1), getMoneyAmountHelper(dp, k + 1, j));
			res = Integer.min(res, tmp);
		}

		dp[i][j] = res;
		return res;
	}

	public int wiggleMaxLength(int[] nums) {
		if (nums.length <= 1)
			return nums.length;

		int k = 0;
		while (k + 1 < nums.length && nums[k] == nums[k + 1])
			k++;

		if (nums.length - k <= 2)
			return nums.length - k;

		int prev = nums[k + 1];
		boolean small = nums[k] < nums[k + 1];
		int res = 2;

		for (int i = k + 2; i < nums.length; i++) {
			if ((nums[i] < prev) == small && nums[i] != prev) {
				res++;
				small = !small;
			}
			prev = nums[i];
		}

		return res;

	}

	public int numDecodings(String s) {
		if (s.length() == 0)
			return 0;
		if (s.charAt(0) == '0')
			return 0;
		if (s.length() == 1)
			return 1;

		int[] dp = new int[s.length()];
		char[] chars = s.toCharArray();

		dp[0] = 1;
		if (chars[1] != '0' && (chars[0] - '0') * 10 + chars[1] - '0' <= 26)
			dp[1] = 2;
		else if (chars[1] != '0' || (chars[1] == '0' && chars[0] < '3'))
			dp[1] = 1;
		else
			return 0;

		for (int i = 2; i < chars.length; i++) {
			if (chars[i] == '0' && (chars[i - 1] > '2' || chars[i - 1] == '0'))
				return 0;
			if (chars[i] != '0')
				dp[i] += dp[i - 1];
			if (chars[i - 1] != '0' && (chars[i - 1] - '0') * 10 + chars[i] - '0' <= 26) {
				dp[i] += dp[i - 2];
			}
		}

		return dp[s.length() - 1];
	}

	public int numSquares(int n) {
		if (n == 0)
			return 0;

		int[] dp = new int[n + 1];
		for (int i = 0; i < n + 1; i++) {
			dp[i] = Integer.MAX_VALUE;
		}

		dp[0] = 0;
		dp[1] = 1;
		for (int i = 2; i <= n; i++) {
			for (int j = 1; j * j <= i; j++) {
				dp[i] = Math.min(dp[i], dp[i - j * j] + 1);
			}
		}

		return dp[n];
	}

	public List<TreeNode> generateTrees(int n) {
		if (n == 0)
			return new LinkedList<>();
		return generateTreesHelper(1, n);
	}

	private List<TreeNode> generateTreesHelper(int start, int end) {
		List<TreeNode> res = new LinkedList<>();

		if (start > end)
			res.add(null);

		for (int i = start; i <= end; i++) {
			List<TreeNode> left = generateTreesHelper(start, i - 1);
			List<TreeNode> right = generateTreesHelper(i + 1, end);

			for (TreeNode l : left) {
				for (TreeNode r : right) {
					TreeNode root = new TreeNode(i);
					root.left = l;
					root.right = r;
					res.add(root);
				}
			}
		}

		return res;
	}

	public boolean canIWin(int maxChoosableInteger, int desiredTotal) {
		// check basic situations
		if (maxChoosableInteger > desiredTotal)
			return true;

		int sum = (maxChoosableInteger + 1) * maxChoosableInteger / 2;
		if (sum < desiredTotal)
			return false;

		boolean[] used = new boolean[maxChoosableInteger + 1];

		Map<Integer, Boolean> m = new HashMap<>();

		return canIWinHelper(desiredTotal, used, m);

	}

	private boolean canIWinHelper(int desiredTotal, boolean[] used, Map<Integer, Boolean> m) {
		if (desiredTotal <= 0)
			return false;

		int key = canIWinFormat(used);

		if (!m.containsKey(key)) {
			for (int i = 1; i < used.length; i++) {
				if (!used[i]) {
					used[i] = true;
					if (!canIWinHelper(desiredTotal - i, used, m)) {
						m.put(key, true);
						used[i] = false;
						return true;
					}
					used[i] = false;
				}
			}
			m.put(key, false);
		}

		return m.get(key);
	}

	private int canIWinFormat(boolean[] used) {
		int res = 0;

		for (boolean b : used) {
			res <<= 1;
			if (b)
				res |= 1;
		}

		return res;
	}

	public int findSubstringInWraproundString(String p) {
		if (p.length() == 0)
			return 0;

		int[] dp = new int[26]; // the max length of subarray end with specific
								// character
		char[] c = p.toCharArray();

		int cnt = 1;

		for (int i = 0; i < c.length; i++) {
			if (i > 0 && (c[i] - c[i - 1] == 1 || c[i - 1] - c[i] == 25)) {
				cnt++;
			} else {
				cnt = 1;
			}

			dp[c[i] - 'a'] = Math.max(dp[c[i] - 'a'], cnt);
		}
		char index = c[c.length - 1];
		dp[index - 'a'] = Math.max(dp[index - 'a'], 1);

		int res = 0;

		for (int i : dp) {
			res += i;
		}

		return res;
	}

	public boolean wordBreak(String s, List<String> wordDict) {
		boolean[] f = new boolean[s.length() + 1];
		f[0] = true;

		for (int i = 1; i <= s.length(); i++) {
			for (String str : wordDict) {
				if (i >= str.length()) {
					if (f[i - str.length()]) {
						if (s.substring(i - str.length(), i).equals(str)) {
							f[i] = true;
							break;
						}
					}
				}
			}
		}

		return f[s.length()];
	}

	public int maxProduct(int[] nums) {
		int res = nums[0];
		int imax = nums[0];
		int imin = nums[0];

		for (int i = 1; i < nums.length; i++) {
			if (nums[i] < 0) {
				int tmp = imin;
				imin = imax;
				imax = tmp;
			}

			imax = Integer.max(imax * nums[i], nums[i]);
			imin = Integer.min(imin * nums[i], nums[i]);

			res = Integer.max(res, imax);
		}

		return res;
	}

	public int robII(int[] nums) {
		if (nums.length == 1)
			return nums[0];
		return Math.max(robIIHelper(nums, 0, nums.length - 1), robIIHelper(nums, 1, nums.length));
	}

	private int robIIHelper(int[] nums, int s, int e) {
		int s1 = 0;
		int s2 = 0, tmp;

		for (int i = s; i < e; i++) {
			tmp = s2;
			s2 = s1 + nums[i];
			s1 = Math.max(s1, tmp);
		}

		return Math.max(s1, s2);
	}

	public boolean checkSubarraySum(int[] nums, int k) {
		Map<Integer, Integer> s = new HashMap<>();
		s.put(0, -1);

		int sum = 0;

		for (int i = 0; i < nums.length; i++) {
			sum += nums[i];
			int tmp = sum;
			if (k != 0)
				tmp = sum % k;
			if (s.containsKey(tmp)) {
				if (i - s.get(tmp) > 1)
					return true;
			} else
				s.put(tmp, i);
		}

		return false;
	}

	public int maximalSquare(char[][] matrix) {
		if (matrix.length == 0)
			return 0;

		int row = matrix.length;
		int col = matrix[0].length;
		int[] dp = new int[col];
		int res = 0, pre;

		// initialize dp matrix
		for (int j = 0; j < col; j++) {
			dp[j] = matrix[0][j] - '0';
			res = Math.max(res, dp[j]);
		}

		// dp[i][j] means the maximal square with matrix[i][j] as the right
		// bottom spot
		for (int i = 1; i < row; i++) {
			pre = dp[0];
			dp[0] = matrix[i][0] - '0';
			res = Math.max(res, dp[0]);
			for (int j = 1; j < col; j++) {
				int tmp = dp[j];
				if (matrix[i][j] == '1') {
					dp[j] = Math.min(Math.min(dp[j], dp[j - 1]), pre) + 1;
				} else
					dp[j] = 0;
				res = Math.max(dp[j], res);
				pre = tmp;
			}
		}

		return res * res;
	}

	public int nthUglyNumber(int n) {
		if (n == 0)
			return 0;
		if (n == 1)
			return 1;

		int k2 = 0, k3 = 0, k5 = 0;
		int[] dp = new int[n];
		dp[0] = 1;

		for (int i = 1; i < n; i++) {
			dp[i] = Math.min(Math.min(dp[k2] * 2, dp[k3] * 3), dp[k5] * 5);
			if (dp[i] == dp[k2] * 2)
				k2++;
			if (dp[i] == dp[k3] * 3)
				k3++;
			if (dp[i] == dp[k5] * 5)
				k5++;
		}

		return dp[n - 1];
	}

	public List<Integer> largestDivisibleSubset(int[] nums) {
		if (nums.length == 0)
			return new LinkedList<Integer>();

		Arrays.sort(nums);

		int m = 1;
		int mi = 0;

		int[] T = new int[nums.length];
		int[] parent = new int[nums.length];
		for (int i = 0; i < nums.length; i++) {
			parent[i] = i;
			T[i] = 1;
		}

		for (int i = 0; i < nums.length; i++) {
			for (int j = i + 1; j < nums.length; j++) {
				if (nums[j] % nums[i] == 0 && T[i] + 1 > T[j]) {
					parent[j] = i;
					T[j] = T[i] + 1;
				}

				if (m < T[j]) {
					m = T[j];
					mi = j;
				}
			}
		}

		List<Integer> res = new LinkedList<Integer>();
		for (int i = 0; i < m; i++) {
			res.add(nums[mi]);
			mi = parent[mi];
		}

		return res;
	}

	public int coinChange(int[] coins, int amount) {
		if (amount == 0)
			return 0;

		int[] dp = new int[amount + 1];
		Arrays.fill(dp, amount + 1);
		dp[0] = 0;

		for (int i = 1; i <= amount; i++) {
			for (int j = 0; j < coins.length; j++) {
				if (i >= coins[j]) {
					dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1);
				}
			}
		}

		return dp[amount] > amount ? -1 : dp[amount];
	}

	public int findPaths(int m, int n, int N, int i, int j) {
		int M = 1000000000 + 7;
		int res = 0;

		// initialize 2D dp array
		int[][] dp = new int[m][n];
		dp[i][j] = 1;

		// an updated dp array for every move
		for (int moves = 1; moves <= N; moves++) {
			int[][] tmp = new int[m][n];

			for (int row = 0; row < m; row++) {
				for (int col = 0; col < n; col++) {
					// calculate number of blocks on the boundary
					if (row == 0)
						res = (res + dp[row][col]) % M;
					if (row == m - 1)
						res = (res + dp[row][col]) % M;
					if (col == 0)
						res = (res + dp[row][col]) % M;
					if (col == n - 1)
						res = (res + dp[row][col]) % M;
					// update the dp array
					tmp[row][col] = (((row - 1 >= 0 ? dp[row - 1][col] : 0) + (col - 1 >= 0 ? dp[row][col - 1] : 0)) % M
							+ ((row + 1 < m ? dp[row + 1][col] : 0) + (col + 1 < n ? dp[row][col + 1] : 0)) % M) % M;
				}
			}

			dp = tmp;
		}

		return res;
	}

	public int minDistance(String word1, String word2) {
		int[][] dp = new int[word1.length() + 1][word2.length() + 1];

		for (int i = 0; i <= word1.length(); i++) {
			for (int j = 0; j <= word2.length(); j++) {
				if (i == 0 || j == 0) {
					dp[i][j] = i + j;
				} else if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
					dp[i][j] = dp[i - 1][j - 1];
				} else {
					dp[i][j] = Math.min(dp[i][j - 1], dp[i - 1][j]) + 1;
				}

			}
		}

		return dp[word1.length()][word2.length()];
	}

	public boolean checkRecord(String s) {
		int cntA = 0;
		for (int i = 0; i < s.length(); i++) {
			if (s.charAt(i) == 'A') {
				cntA++;
				if (cntA > 1)
					return false;
			} else if (s.charAt(i) == 'L') {
				int cnt = 1;
				while (i + 1 < s.length() && s.charAt(++i) == 'L')
					cnt++;
				i--;
				if (cnt > 2)
					return false;
				if (i == s.length() - 2)
					break;
			}
		}
		return true;
	}

	// public int[] smallestRange(List<List<Integer>> nums) {
	// int[] cur = new int[nums.size()];
	// int max = Integer.MIN_VALUE;
	//
	// PriorityQueue<Integer> pq = new
	// PriorityQueue<>((i,j)->nums.get(i).get(cur[i]) -
	// nums.get(j).get(cur[j]));
	//
	// for(int i = 0; i < nums.size(); i++) {
	// pq.offer(i);
	// max = Integer.max(max, nums.get(i).get(0));
	// }
	// int minLow = pq.poll();;
	// int minHigh = max;
	// boolean flag = false;
	//
	// while(true) {
	// cur[minLow]++;
	//
	// for(int i = 0; i < nums.size(); i++) {
	// if(cur[i] >= nums.get(i).size()) flag = true;
	// }
	// if(flag) break;
	//
	// pq.offer(minLow);
	// max = Integer.max(max, nums.get(minLow).get(cur[minLow]));
	// int tmpLow = pq.poll();
	// if(minHigh - nums.get(minLow).get(cur[minLow]) > max -
	// nums.get(tmpLow).get(cur[tmpLow])) {
	// minHigh = max;
	// minLow = tmpLow;
	// }
	// }
	//
	// return new int[]{nums.get(minLow).get(cur[minLow]), minHigh};
	// }

	public String reverseVowels(String s) {
		String vowels = "aeiouAEIOU";

		int low = 0;
		int high = s.length() - 1;
		char[] chars = s.toCharArray();

		while (low < high) {
			while (low < high && !vowels.contains(chars[low] + "")) {
				low++;
			}
			while (low < high && !vowels.contains(chars[high] + "")) {
				high--;
			}
			char tmp = chars[low];
			chars[low] = chars[high];
			chars[high] = tmp;
			low++;
			high--;
		}

		return new String(chars);
	}

	public boolean repeatedSubstringPattern(String s) {
		int l = s.length();

		for (int i = l / 2; i > 0; i--) {
			if (l % i == 0) {
				int m = l / i;
				String rep = s.substring(0, i);
				StringBuilder strb = new StringBuilder();

				for (int j = 0; j < m; j++) {
					strb.append(rep);
				}

				if (strb.toString().equals(s))
					return true;
			}
		}

		return false;
	}

	public int strStr(String haystack, String needle) {
		for (int i = 0;; i++) {
			for (int j = 0;; j++) {
				if (j == needle.length())
					return i;
				if (i + j == haystack.length())
					return -1;
				if (haystack.charAt(i + j) != needle.charAt(j))
					break;
			}
		}
	}

	public int countSegments(String s) {
		int res = 0;

		for (int i = 0; i < s.length(); i++) {
			if (s.charAt(i) != ' ' && (i == 0 || s.charAt(i - 1) == ' '))
				res++;
		}

		return res;
	}

	public String countAndSay(int n) {
		StringBuilder cur;
		StringBuilder prev = new StringBuilder("1");

		for (int i = 1; i < n; i++) {
			cur = new StringBuilder();
			char say = prev.charAt(0);
			int cnt = 1;
			for (int j = 1; j < prev.length(); j++) {
				if (prev.charAt(j) == say)
					cnt++;
				else {
					cur.append(cnt).append(say);
					cnt = 1;
					say = prev.charAt(j);
				}
			}
			cur.append(cnt).append(say);
			prev = cur;
		}

		return prev.toString();
	}

	public List<String> letterCombinations(String digits) {
		String[] dic = new String[] { "0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz" };
		List<String> res = new LinkedList<>();
		if (digits.length() == 0)
			return res;
		res.add("");

		for (int i = 0; i < digits.length(); i++) {
			int x = digits.charAt(i) - '0';
			if (x < 2)
				return new LinkedList<>();
			String str = dic[x];
			while (res.get(0).length() == i) {
				String prev = res.get(0);
				for (int j = 0; j < str.length(); j++) {
					res.add(prev + str.charAt(j));
				}
				res.remove(0);
			}
		}

		return res;
	}

	public List<List<String>> groupAnagrams(String[] strs) {
		if (strs.length == 0)
			return new LinkedList<>();

		Map<String, List<String>> m = new HashMap<>();

		for (String str : strs) {
			char[] c = str.toCharArray();
			Arrays.sort(c);
			String s = new String(c);
			if (!m.containsKey(s))
				m.put(s, new LinkedList<>());
			m.get(s).add(str);
		}

		return new LinkedList<>(m.values());

	}

	public boolean isValid(String s) {
		Stack<Character> stack = new Stack<>();

		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (c == '(')
				stack.push(')');
			else if (c == '[')
				stack.push(']');
			else if (c == '{')
				stack.push('}');
			else {
				if (stack.isEmpty() || stack.pop() != c)
					return false;
			}
		}

		return stack.isEmpty();
	}

	public String addBinary(String a, String b) {
		int m = a.length();
		int n = b.length();
		if (m > n)
			return addBinary(b, a);
		if (m == 0)
			a = "0";
		if (n == 0)
			b = "0";

		char[] chb = b.toCharArray();

		int c = 0;
		for (int i = 1; i <= m; i++) {
			int ca = a.charAt(m - i) - '0';
			int cb = chb[n - i] - '0';
			chb[n - i] = (char) (ca ^ cb ^ c + '0');
			if (ca + cb + c > 1)
				c = 1;
			else
				c = 0;
		}

		if (c == 1) {
			for (int i = m + 1; i <= n && c == 1; i++) {
				int cb = chb[n - i] - '0';
				chb[n - i] = (char) (cb ^ c + '0');
				if (cb + c > 1)
					c = 1;
				else
					c = 0;
			}

			if (c == 1)
				return 1 + new String(chb);
		}

		return new String(chb);
	}

	public int lengthOfLastWord(String s) {
		int cnt = 0;
		boolean flag = false;

		for (int i = s.length() - 1; i >= 0; i--) {
			char c = s.charAt(i);
			if (c != ' ') {
				cnt++;
				flag = true;
			}
			if (flag && s.charAt(i) == ' ')
				break;
		}

		return cnt;
	}

	public String longestCommonPrefix(String[] strs) {
		if (strs.length == 0)
			return "";
		int res = 0;
		StringBuilder sb = new StringBuilder("");

		while (true) {
			if (res == strs[0].length())
				return sb.toString();
			char c = strs[0].charAt(res);
			for (int i = 1; i < strs.length; i++) {
				if (strs[i].length() == res || strs[i].charAt(res) != c)
					return sb.toString();
			}
			sb.append(c);
			res++;
		}
	}

	public int calculate(String s) {
		Stack<Integer> stack = new Stack<>();

		int num = 0;
		char sign = '+';
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (Character.isDigit(c)) {
				num = num * 10 + c - '0';
			}
			if ((!Character.isDigit(c) && c != ' ') || i == s.length() - 1) {
				if (sign == '+') {
					stack.push(num);
				} else if (sign == '-') {
					stack.push(-num);
				} else if (sign == '*') {
					stack.push(stack.pop() * num);
				} else if (sign == '/') {
					stack.push(stack.pop() / num);
				}
				sign = c;
				num = 0;
			}
		}

		num = 0;

		while (!stack.isEmpty()) {
			num += stack.pop();
		}

		return num;
	}

	public int nextGreaterElement(int n) {
		char[] ch = (n + "").toCharArray();
		int i;
		for (i = ch.length - 1; i > 0; i--) {
			if (ch[i - 1] < ch[i]) {
				break;
			}
		}

		if (i == 0)
			return -1;

		int smallest = i;
		for (int j = i; j < ch.length; j++) {
			if (ch[j] > ch[i - 1] && ch[j] < ch[smallest]) {
				smallest = j;
			}
		}
		char tmp = ch[smallest];
		ch[smallest] = ch[i - 1];
		ch[i - 1] = tmp;

		Arrays.sort(ch, i, ch.length);

		long res = Long.parseLong(new String(ch));
		return res < Integer.MAX_VALUE ? (int) res : -1;
	}

	public List<String> restoreIpAddresses(String s) {
		List<String> res = new LinkedList<>();

		if (s.length() > 12)
			return res;

		for (int i = 0; i < s.length() - 2; i++) {
			for (int j = i + 1; j < s.length() - 1; j++) {
				for (int k = j + 1; k < s.length(); k++) {
					String s1 = s.substring(0, i);
					String s2 = s.substring(i, j);
					String s3 = s.substring(j, k);
					String s4 = s.substring(k);
					if (isValidIpAddresses(s1) && isValidIpAddresses(s2) && isValidIpAddresses(s3)
							&& isValidIpAddresses(s4)) {
						String str = new String(s1 + '.' + s2 + '.' + s3 + '.' + s4);
						res.add(str);
					}
				}
			}
		}

		return res;

	}

	private boolean isValidIpAddresses(String s) {
		if (s.length() > 3 || s.length() == 0 || (s.charAt(0) == '0' && s.length() > 1) || Integer.parseInt(s) > 255) {
			return false;
		}
		return true;
	}

	public String multiply(String num1, String num2) {
		int m = num1.length();
		int n = num2.length();
		int[] res = new int[n + m];

		for (int i = m - 1; i >= 0; i--) {
			for (int j = n - 1; j >= 0; j--) {
				int a = (num1.charAt(i) - '0') * (num2.charAt(j) - '0');
				int sum = res[i + j + 1] + (a % 10);
				res[i + j + 1] = sum % 10;
				res[i + j] += a / 10 + sum / 10;
			}
		}

		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < m + n; i++) {
			if (!(res[i] == 0 && sb.length() == 0))
				sb.append(res[i]);
		}

		return sb.length() == 0 ? "0" : sb.toString();
	}

	public String convert(String s, int numRows) {
		StringBuilder[] sb = new StringBuilder[numRows];
		for (int i = 0; i < numRows; i++)
			sb[i] = new StringBuilder();

		int i = 0;
		int len = s.length();
		while (i < len) {
			for (int row = 0; row < numRows && i < len; row++) {
				sb[row].append(s.charAt(i++));
			}

			for (int row = numRows - 2; row > 0 && i < len; row--) {
				sb[row].append(s.charAt(i++));
			}
		}

		for (int ind = 1; ind < numRows; ind++) {
			sb[0].append(sb[ind]);
		}

		return sb[0].toString();
	}

	public boolean isPalindrome(String s) {
		int i = 0;
		int j = s.length() - 1;
		s = s.toLowerCase();

		while (i < j) {
			while (!Character.isAlphabetic(s.charAt(i)) && !Character.isDigit(s.charAt(i)) && i < j) {
				i++;
			}
			while ((!Character.isAlphabetic(s.charAt(j))) && !Character.isDigit(s.charAt(j)) && i < j) {
				j--;
			}

			if (s.charAt(i) != s.charAt(j))
				return false;
			i++;
			j--;

		}

		return true;
	}

	public boolean isScramble(String s1, String s2) {
		if (s1.equals(s2))
			return true;

		int[] cnt = new int[26];
		int len = s1.length();

		for (int i = 0; i < len; i++) {
			cnt[s1.charAt(i) - 'a']++;
			cnt[s2.charAt(i) - 'a']--;
		}

		for (int i = 0; i < 26; i++) {
			if (cnt[i] != 0)
				return false;
		}

		for (int i = 0; i < len; i++) {
			if (isScramble(s1.substring(0, i), s2.substring(0, i)) && isScramble(s1.substring(i), s2.substring(i))) {
				return true;
			}
			if (isScramble(s1.substring(0, i), s2.substring(len - i))
					&& isScramble(s1.substring(i), s2.substring(0, len - i))) {
				return true;
			}

		}

		return false;
	}

	public int myAtoi(String str) {
		if (str.length() == 0)
			return 0;

		int sign = 1;
		int num = 0;

		int start = 0;
		while (start < str.length() && str.charAt(start) == ' ')
			start++;
		str = str.substring(start);

		char ch = str.charAt(0);
		if (ch == '-') {
			sign = -1;
			str = str.substring(1);
		} else if (ch == '+') {
			str = str.substring(1);
		} else if (!Character.isDigit(ch)) {
			return 0;
		}

		for (int i = 0; i < str.length(); i++) {
			if (!Character.isDigit(str.charAt(i))) {
				str = str.substring(0, i);
				break;
			}
		}

		start = 0;
		while (start < str.length() && str.charAt(start) == '0')
			start++;
		str = str.substring(start);

		if (str.equals("2147483648") && sign < 0)
			return Integer.MIN_VALUE;

		for (int i = 0; i < str.length(); i++) {
			num = num * 10 + str.charAt(i) - '0';

			if (num < 0 || i > 9) {
				if (sign > 0)
					return Integer.MAX_VALUE;
				else
					return Integer.MIN_VALUE;
			}

		}

		return sign * num;
	}

	public boolean isInterleave(String s1, String s2, String s3) {
		if (s1.length() + s2.length() != s3.length())
			return false;

		if (s1.length() == 0)
			return s3.equals(s2);
		if (s2.length() == 0)
			return s3.equals(s1);

		boolean[][] dp = new boolean[s1.length() + 1][s2.length() + 1];

		dp[0][0] = true;

		for (int i = 1; i <= s1.length(); i++) {
			if (s1.charAt(i - 1) == s3.charAt(i - 1))
				dp[i][0] = dp[i - 1][0] && true;
		}

		for (int j = 1; j <= s2.length(); j++) {
			if (s2.charAt(j - 1) == s3.charAt(j - 1))
				dp[0][j] = dp[0][j - 1] && true;
		}

		for (int i = 1; i <= s1.length(); i++) {
			for (int j = 1; j <= s2.length(); j++) {
				char ch1 = s1.charAt(i - 1);
				char ch2 = s2.charAt(j - 1);
				char ch3 = s3.charAt(i + j - 1);
				dp[i][j] = (ch2 == ch3 && dp[i][j - 1]) || (ch1 == ch3 && dp[i - 1][j]);
			}
		}

		return dp[s1.length()][s2.length()];
	}

	public String minWindow(String s, String t) {
		Map<Character, Integer> map = new HashMap<>();

		for (int i = 0; i < t.length(); i++) {
			char c = t.charAt(i);
			if (map.containsKey(c))
				map.put(c, map.get(c) + 1);
			else
				map.put(c, 1);
		}

		int start = 0, end = 0;
		int minStart = 0;
		int minLen = Integer.MAX_VALUE;
		int cnt = t.length();

		while (end < s.length()) {
			if (map.containsKey(s.charAt(end))) {
				if (map.get(s.charAt(end)) > 0)
					cnt--; // if it is needed in the window, else it is in the
							// string t, but not needed now
				map.put(s.charAt(end), map.get(s.charAt(end)) - 1);
			}
			end++; // move end to make the window valid
			while (cnt == 0) { // move start to make the window smaller and
								// invalid again
				if (end - start < minLen) {
					minStart = start;
					minLen = end - start;
				}

				if (map.containsKey(s.charAt(start))) {
					if (map.get(s.charAt(start)) == 0)
						cnt++;// if it can make the window invalid
					map.put(s.charAt(start), map.get(s.charAt(start)) + 1);
				}

				start++;
			}
		}

		return minLen > s.length() ? "" : s.substring(minStart, minLen + minStart);
	}

	public boolean isMatch(String s, String p) {
		boolean[][] dp = new boolean[s.length() + 1][p.length() + 1];

		dp[0][0] = true;

		for (int i = 0; i < p.length(); i++) {
			if (p.charAt(i) == '*' && dp[0][i - 1])
				dp[0][i + 1] = true;
		}

		for (int i = 1; i <= s.length(); i++) {
			for (int j = 1; j <= p.length(); j++) {
				if (p.charAt(j - 1) == '.') {
					dp[i][j] = dp[i - 1][j - 1];
				}
				if (p.charAt(j - 1) == s.charAt(i - 1)) {
					dp[i][j] = dp[i - 1][j - 1];
				}
				if (p.charAt(j - 1) == '*') {
					if (s.charAt(i - 1) != p.charAt(j - 2) && p.charAt(j - 2) != '.') {
						dp[i][j] = dp[i][j - 2];
					} else {
						dp[i][j] = dp[i][j - 1] || dp[i - 1][j] || dp[i][j - 2];
					}
				}
			}
		}

		return dp[s.length()][p.length()];
	}

	public String fractionAddition(String expression) {
		List<List<Integer>> fractions = new ArrayList<>();
		List<Integer> sign = new ArrayList<>();

		if (expression.charAt(0) == '-')
			sign.add(-1);
		else
			sign.add(1);

		for (int i = 1; i < expression.length(); i++) {
			if (expression.charAt(i) == '+') {
				sign.add(1);
			} else if (expression.charAt(i) == '-') {
				sign.add(-1);
			}
		}
		int i = 0;

		for (String fraction : expression.split("-|\\+")) {
			if (fraction.length() > 0) {
				String[] nums = fraction.split("/");
				int a = Integer.parseInt(nums[0]);
				int b = Integer.parseInt(nums[1]);
				fractions.add(new ArrayList<Integer>());
				fractions.get(i).add(a * sign.get(i));
				fractions.get(i).add(b);
				i++;
			}
		}

		List<Integer> res = fractions.get(0);

		for (int j = 1; j < fractions.size(); j++) {
			List<Integer> cur = fractions.get(j);
			int den = res.get(1) * cur.get(1);
			int num = res.get(0) * cur.get(1) + res.get(1) * cur.get(0);
			int lcm = getLCM(den, num);
			res.set(0, num / lcm);
			res.set(1, den / lcm);
		}

		return res.get(0) + "/" + res.get(1);

	}

	private int getLCM(int a, int b) {
		if (a < b)
			return getLCM(b, a);

		if (b == 0)
			return Math.abs(a);

		while (a % b != 0) {
			int c = a % b;
			a = b;
			b = c;
		}

		return Math.abs(b);
	}

	public int maxCount(int m, int n, int[][] ops) {
		int[] res = new int[] { Integer.MAX_VALUE, Integer.MAX_VALUE };

		for (int i = 0; i < ops.length; i++) {
			res[0] = Math.min(res[0], ops[i][0]);
			res[1] = Math.min(res[1], ops[i][1]);
		}

		return (res[0] < m ? res[0] : m) * (res[1] < n ? res[1] : n);
	}

	public int[] findErrorNums(int[] nums) {
		int[] ind = new int[nums.length];
		int[] res = new int[2];

		for (int i : nums) {
			if (ind[i - 1] == 1)
				res[0] = i;
			ind[i - 1] = 1;
		}

		for (int i = 0; i < ind.length; i++) {
			if (ind[i] == 0) {
				res[1] = i + 1;
				break;
			}
		}

		return res;
	}

	public boolean isHappy(int n) {
		Set<Integer> set = new HashSet<>();
		int res = n;

		while (res != 1) {
			n = res;
			res = 0;

			while (n > 0) {
				res += (n % 10) * (n % 10);
				n = n / 10;
			}

			if (res == 1)
				return true;
			if (!set.add(res))
				return false;
		}

		return true;
	}

	public boolean isPowerOfTwo(int n) {
		return n > 0 && (n & (n - 1)) == 0;
	}

	public boolean isPowerOfThree(int n) {
		return n > 0 && 1162261467 % n == 0;
	}

	public boolean isPowerOfFour(int n) {
		return n > 0 && (n & (n - 1)) == 0 && (n & 0x55555555) == 0;
	}

	public String replaceWords(List<String> dict, String sentence) {
		Trie trie = new Trie(256);
		dict.forEach(str -> trie.insert(str));
		List<String> res = new LinkedList<>();
		Arrays.stream(sentence.split(" ")).forEach(str -> res.add(trie.getShortestPre(str)));
		return res.stream().collect(Collectors.joining(" "));
	}

	public int leastBricks(List<List<Integer>> wall) {

		Map<Integer, Integer> edges = new HashMap<>();

		for (List<Integer> row : wall) {
			int cur = 0;
			for (int i = 0; i < row.size() - 1; i++) {
				cur += row.get(i);
				edges.put(cur, edges.getOrDefault(cur, 0) + 1);
			}
		}
		int max = 0;
		for (Map.Entry<Integer, Integer> e : edges.entrySet()) {
			max = Math.max(max, e.getValue());
		}

		return wall.size() - max;
	}

	public int findLHS(int[] nums) {
		Map<Integer, Integer> occur = new HashMap<>();

		for (int cur : nums)
			occur.put(cur, occur.getOrDefault(cur, 0) + 1);

		int max = 0;
		for (int key : occur.keySet()) {
			if (occur.containsKey(key + 1))
				max = Math.max(max, occur.get(key) + occur.get(key + 1));
		}

		return max;
	}

	public int findMaxLength(int[] nums) {
		Map<Integer, Integer> map = new HashMap<>();

		int cnt = 0;
		int res = 0;

		map.put(0, 0);

		for (int i = 0; i < nums.length; i++) {
			if (nums[i] == 0)
				cnt--;
			else
				cnt++;
			if (map.containsKey(cnt))
				res = Math.max(res, i - map.get(cnt) + 1);
			else
				map.put(cnt, i + 1);
		}

		return res;
	}

	public boolean isValidSudoku(char[][] board) {
		boolean[][] usedRow = new boolean[9][9];
		boolean[][] usedCol = new boolean[9][9];
		boolean[][] usedBox = new boolean[9][9];

		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 9; j++) {
				if (board[i][j] != '.') {
					int k = i / 3 * 3 + j / 3;
					int num = board[i][j] - '0' - 1;
					if (usedRow[i][num] || usedCol[j][num] || usedBox[k][num])
						return false;
					usedRow[i][num] = true;
					usedCol[j][num] = true;
					usedBox[k][num] = true;
				}
			}
		}

		return true;
	}

	public String getHint(String secret, String guess) {
		int As = 0;
		int Bs = 0;
		int[] digits = new int[10];

		for (int i = 0; i < secret.length(); i++) {
			if (secret.charAt(i) == guess.charAt(i))
				As++;
			digits[secret.charAt(i) - '0']++;
		}

		for (int i = 0; i < guess.length(); i++) {
			if (digits[guess.charAt(i) - '0'] > 0) {
				Bs++;
				digits[guess.charAt(i) - '0']--;
			}
		}

		return As + "A" + (Bs - As) + "B";
	}

	public List<Double> averageOfLevels(TreeNode root) {
		List<Double> res = new LinkedList<>();

		Queue<TreeNode> queue = new LinkedList<>();

		queue.add(root);

		while (!queue.isEmpty()) {
			int size = queue.size();
			double ave = 0;
			for (int i = 0; i < size; i++) {
				TreeNode cur = queue.poll();
				ave += cur.val;
				if (cur.left != null)
					queue.add(cur.left);
				if (cur.right != null)
					queue.add(cur.right);
			}
			ave = ave / size;
			res.add(ave);
		}

		return res;
	}

	public TreeNode addOneRow(TreeNode root, int v, int d) {
		if (d == 1) {
			TreeNode newRoot = new TreeNode(v);
			newRoot.left = root;
			return newRoot;
		}

		Queue<TreeNode> queue = new LinkedList<>();

		queue.add(root);

		int depth = 1;

		while (!queue.isEmpty()) {
			if (depth < d - 1) {
				int size = queue.size();
				for (int i = 0; i < size; i++) {
					TreeNode cur = queue.poll();
					if (cur.left != null)
						queue.add(cur.left);
					if (cur.right != null)
						queue.add(cur.right);
				}
				depth++;
			} else {
				int size = queue.size();
				for (int i = 0; i < size; i++) {
					TreeNode cur = queue.poll();
					TreeNode left = new TreeNode(v);
					left.left = cur.left;
					cur.left = left;
					TreeNode right = new TreeNode(v);
					right.right = cur.right;
					cur.right = right;
				}
				break;
			}
		}

		return root;
	}

	private int diameterOfBinaryTreeMax = 0;

	public int diameterOfBinaryTree(TreeNode root) {
		diameterOfBinaryTreeHelper(root);

		return diameterOfBinaryTreeMax;
	}

	private int diameterOfBinaryTreeHelper(TreeNode root) {
		if (root == null)
			return 0;

		int maxLeft = diameterOfBinaryTreeHelper(root.left);
		int maxRight = diameterOfBinaryTreeHelper(root.right);

		diameterOfBinaryTreeMax = Math.max(diameterOfBinaryTreeMax, maxLeft + maxRight + 1);

		return Math.max(maxLeft, maxRight) + 1;
	}

	public boolean isSubtree(TreeNode s, TreeNode t) {
		return isSubtreeTraverse(s, t);
	}

	private boolean isSubtreeEquals(TreeNode s, TreeNode t) {
		if (s == null && t == null)
			return true;
		if (s == null || t == null)
			return false;

		return s.val == t.val && isSubtreeEquals(s.left, t.left) && isSubtreeEquals(s.right, t.right);
	}

	private boolean isSubtreeTraverse(TreeNode s, TreeNode t) {
		return s != null && (isSubtreeEquals(s, t) || isSubtreeTraverse(s.left, t) || isSubtreeTraverse(s.right, t));
	}

	public List<Integer> rightSideView(TreeNode root) {
		List<Integer> res = new LinkedList<>();

		if (root == null)
			return res;

		Queue<TreeNode> queue = new LinkedList<>();
		queue.add(root);

		while (!queue.isEmpty()) {
			int size = queue.size();
			TreeNode cur = queue.peek();
			for (int i = 0; i < size; i++) {
				cur = queue.poll();
				if (cur.left != null)
					queue.add(cur.left);
				if (cur.right != null)
					queue.add(cur.right);
			}

			res.add(cur.val);
		}

		return res;

	}

	public List<Integer> postorderTraversal(TreeNode root) {
		Stack<TreeNode> stack = new Stack<>();
		List<Integer> res = new LinkedList<>();

		if (root == null)
			return res;

		stack.push(root);

		while (!stack.isEmpty()) {
			TreeNode p = stack.pop();
			res.add(p.val);
			if (p.left != null)
				stack.push(p.left);
			if (p.right != null)
				stack.push(p.right);
		}

		Collections.reverse(res);

		return res;
	}

	public List<List<Integer>> levelOrderBottom(TreeNode root) {
		List<List<Integer>> res = new ArrayList<>();

		if (root == null)
			return res;

		Queue<TreeNode> queue = new LinkedList<>();
		queue.add(root);

		while (!queue.isEmpty()) {
			List<Integer> list = new LinkedList<>();
			int size = queue.size();

			for (int i = 0; i < size; i++) {
				TreeNode cur = queue.poll();
				if (cur.left != null)
					queue.add(cur.left);
				if (cur.right != null)
					queue.add(cur.right);

				list.add(cur.val);
			}

			res.add(0, list);
		}

		return res;

	}

	public int pathSum(TreeNode root, int sum) {
		Map<Integer, Integer> map = new HashMap<>();
		map.put(0, 1);

		return pathSumHelper(root, 0, sum, map);
	}

	private int pathSumHelper(TreeNode root, int curSum, int target, Map<Integer, Integer> map) {
		if (root == null)
			return 0;

		curSum += root.val;
		int res = map.getOrDefault(curSum - target, 0);
		map.put(curSum, map.getOrDefault(curSum, 0) + 1);

		res += pathSumHelper(root.left, curSum, target, map) + pathSumHelper(root.right, curSum, target, map);
		map.put(curSum, map.get(curSum) - 1);

		return res;
	}

	public List<List<Integer>> levelOrder(TreeNode root) {
		List<List<Integer>> res = new ArrayList<>();

		if (root == null)
			return res;

		Queue<TreeNode> queue = new LinkedList<>();
		queue.add(root);

		while (!queue.isEmpty()) {
			List<Integer> list = new LinkedList<>();
			int size = queue.size();

			for (int i = 0; i < size; i++) {
				TreeNode cur = queue.poll();
				if (cur.left != null)
					queue.add(cur.left);
				if (cur.right != null)
					queue.add(cur.right);

				list.add(cur.val);
			}

			res.add(list);
		}

		return res;
	}

	public TreeNode lowestCommonAncestorBST(TreeNode root, TreeNode p, TreeNode q) {
		while ((p.val - root.val) * (q.val - root.val) > 0)
			root = (p.val - root.val) > 0 ? root.right : root.left;
		return root;
	}

	public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
		if (root == null || root == p || root == q)
			return root;

		TreeNode left = lowestCommonAncestor(root.left, p, q);
		TreeNode right = lowestCommonAncestor(root.right, p, q);

		if (left == null)
			return right;
		else if (right == null)
			return left;
		else
			return root;

	}

	public void recoverTree(TreeNode root) {
		TreeNode cur = root;
		TreeNode prev = null;
		TreeNode tmp = null;
		TreeNode first = null, second = null;

		while (cur != null) {

			if (cur.left != null) {
				tmp = cur.left;
				// find the precedence of cur
				while (tmp.right != null && tmp.right != cur)
					tmp = tmp.right;

				if (tmp.right == null) {
					tmp.right = cur;
					cur = cur.left;
				} else {
					if (prev != null && cur.val < prev.val) {
						if (first == null)
							first = prev;
						second = cur;
					}
					prev = cur;
					tmp.right = null;
					cur = cur.right;
				}
			} else {
				if (prev != null && cur.val < prev.val) {
					if (first == null)
						first = prev;
					second = cur;
				}
				prev = cur;
				cur = cur.right;

			}
		}

		if (first != null && second != null) {
			int swap = first.val;
			first.val = second.val;
			second.val = swap;
		}
	}

	public List<List<Integer>> findSubsequences(int[] nums) {
		Set<List<Integer>> res = new HashSet<>();

		findSubsequences(res, new LinkedList<>(), 0, nums);

		return new LinkedList<>(res);
	}

	private void findSubsequences(Set<List<Integer>> res, List<Integer> cur, int ind, int[] nums) {
		if (cur.size() > 1)
			res.add(new LinkedList<>(cur));

		for (int i = ind; i < nums.length; i++) {
			if (cur.size() == 0 || nums[i] >= cur.get(cur.size() - 1)) {
				cur.add(nums[i]);
				findSubsequences(res, cur, i + 1, nums);
				cur.remove(cur.size() - 1);
			}
		}

		return;
	}

	public boolean isSymmetric(TreeNode root) {
		return root == null || isSymmetricHelper(root.left, root.right);
	}

	private boolean isSymmetricHelper(TreeNode left, TreeNode right) {
		if (left == null || right == null)
			return left == right;

		if (left.val != right.val)
			return false;
		else {
			return isSymmetricHelper(left.left, right.right) && isSymmetricHelper(left.right, right.left);
		}
	}

	public List<String> binaryTreePaths(TreeNode root) {
		List<String> res = new LinkedList<>();

		if (root == null)
			return res;

		binaryTreePathsHelper(root, res, "" + root.val);

		return res;
	}

	private void binaryTreePathsHelper(TreeNode root, List<String> res, String sb) {
		if (root.left == null && root.right == null) {
			res.add(new String(sb));
			return;
		}

		sb += "->";

		if (root.left != null)
			binaryTreePathsHelper(root.left, res, sb + root.left.val);
		if (root.right != null)
			binaryTreePathsHelper(root.right, res, sb + root.right.val);
	}

	public ListNode addTwoNumbers(ListNode l1, ListNode l2) {

		ListNode p = l1;
		ListNode q = l2;
		int c = 0;
		ListNode res = new ListNode(0);
		ListNode r = res;

		while (p != null || q != null) {
			int x = (p == null) ? 0 : p.val;
			int y = (q == null) ? 0 : q.val;
			int sum = x + y + c;
			c = sum / 10;
			sum = sum % 10;
			r.next = new ListNode(sum);
			if (p != null)
				p = p.next;
			if (q != null)
				q = q.next;
			r = r.next;
		}

		if (c > 0)
			r.next = new ListNode(c);

		return res.next;
	}

	public int lengthOfLongestSubstring(String s) {
		if (s.length() == 0)
			return 0;

		Map<Character, Integer> m = new HashMap<>();
		m.put(s.charAt(0), 1);

		int i = 0, j = 1;
		int max = 1;

		while (i <= j && j < s.length()) {
			m.put(s.charAt(j), m.getOrDefault(s.charAt(j), 0) + 1);
			if (m.get(s.charAt(j)) <= 1) {
				j++;
				max = Math.max(max, j - i);
			} else {
				while (m.get(s.charAt(i)) == 1) {
					m.put(s.charAt(i), 0);
					i++;
				}
				m.put(s.charAt(i), 1);
				i++;
				j++;
			}
		}

		return max;
	}

	public double findMedianSortedArrays(int[] nums1, int[] nums2) {
		if (nums1.length > nums2.length)
			return findMedianSortedArrays(nums2, nums1);

		int m = nums1.length, n = nums2.length;
		int low = 0, high = m;

		while (low <= high) {
			int mid1 = (high - low) / 2 + low;
			int mid2 = (m + n) / 2 - mid1;

			double l1 = (mid1 == 0) ? Integer.MIN_VALUE : nums1[mid1];
			double l2 = (mid2 == 0) ? Integer.MIN_VALUE : nums2[mid2];
			double r1 = (mid1 == m - 1) ? Integer.MAX_VALUE : nums1[mid1 + 1];
			double r2 = (mid2 == n - 1) ? Integer.MAX_VALUE : nums2[mid2 + 1];

			if (l1 <= r2 && r1 >= l2)
				return (Math.max(l1, l2) + Math.min(r1, r2)) / 2.0;
			else if (l1 > r2) {
				high = mid1 - 1;
			} else {
				low = mid1 + 1;
			}
		}

		return -1;

	}

	public String longestPalindrome(String s) {
		if (s.length() == 0)
			return "";

		int start = 0, end = 0;

		for (int i = 0; i < s.length(); i++) {
			int len1 = longestPalindromeHelper(s, i, i);
			int len2 = longestPalindromeHelper(s, i, i + 1);

			int len = Math.max(len1, len2);

			if (len > end - start) {
				start = i - (len - 1) / 2;
				end = i + len / 2;
			}
		}

		return s.substring(start, end + 1);
	}

	private int longestPalindromeHelper(String s, int l, int r) {
		while (l >= 0 && r < s.length() && s.charAt(l) == s.charAt(r)) {
			l--;
			r++;
		}

		return r - l - 1;
	}

	public int reverse(int x) {
		int sign = x >= 0 ? 1 : -1;
		int res = 0;
		x = Math.abs(x);

		while (x > 0) {
			if (res > 214748365)
				return 0;
			res = res * 10 + x % 10;
			if (res < 0)
				return 0;
			x = x / 10;
		}

		if (res == Integer.MAX_VALUE && sign == -1)
			return 0;
		return res * sign;

	}

	public boolean isPalindrome(int x) {
		if (x < 0 || (x > 0 && x % 10 == 0))
			return false;
		int rev = 0;

		while (x > rev) {
			rev = rev * 10 + x % 10;
			x = x / 10;
		}

		return x == rev || x == rev / 10;
	}

	public List<List<Integer>> threeSum(int[] nums) {
		List<List<Integer>> res = new ArrayList<>();

		Arrays.sort(nums);

		for (int i = 0; i < nums.length - 2; i++) {
			int low = i + 1;
			int high = nums.length - 1;
			int target = 0 - nums[i];
			while (low < high) {
				if (nums[low] + nums[high] == target) {
					res.add(Arrays.asList(nums[i], nums[low], nums[high]));
					while (low < high && nums[low] == nums[low + 1])
						low++;
					while (low < high && nums[high] == nums[high - 1])
						high--;
					low++;
					high--;
				} else if (nums[low] + nums[high] < target)
					low++;
				else
					high--;
			}
			while (i < nums.length - 2 && nums[i] == nums[i + 1])
				i++;
		}

		return res;
	}

	public ListNode removeNthFromEnd(ListNode head, int n) {
		ListNode p = head;
		ListNode q = head;
		for (int i = 0; i < n; i++)
			q = q.next;
		if (q == null)
			return head.next;

		while (q.next != null) {
			p = p.next;
			q = q.next;
		}

		p.next = p.next.next;

		return head;
	}

	public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
		ListNode res = new ListNode(0);
		ListNode p = res;

		while (l1 != null && l2 != null) {
			if (l1.val < l2.val) {
				ListNode tmp = new ListNode(l1.val);
				p.next = tmp;
				l1 = l1.next;
			} else {
				ListNode tmp = new ListNode(l2.val);
				p.next = tmp;
				l2 = l2.next;
			}

			p = p.next;
		}

		if (l1 == null)
			p.next = l2;
		else
			p.next = l1;

		return res.next;
	}

	public ListNode mergeKLists(ListNode[] lists) {
		ListNode res = new ListNode(0);
		ListNode p = res;

		PriorityQueue<ListNode> pq = new PriorityQueue<>((i, j) -> i.val - j.val);

		for (ListNode i : lists)
			if (i != null)
				pq.add(i);

		while (!pq.isEmpty()) {
			p.next = pq.poll();
			p = p.next;
			if (p.next != null)
				pq.add(p.next);
		}

		return res.next;

	}

	public ListNode swapPairs(ListNode head) {
		if (head == null)
			return head;

		ListNode dummy = new ListNode(0);

		dummy.next = head;
		ListNode p = dummy;
		ListNode q = head;
		ListNode r = head.next;

		while (r != null) {
			p.next = r;
			p = q;
			q.next = r.next;
			r.next = q;
			q = q.next;
			if (q != null)
				r = q.next;
			else
				break;
		}

		return dummy.next;
	}

	public ListNode reverseKGroup(ListNode head, int k) {
		ListNode p = head;
		int cnt = 0;
		while (p != null && cnt != k) {
			p = p.next;
			cnt++;
		}

		if (cnt == k) {
			p = reverseKGroup(p, k);

			while (cnt > 0) {
				ListNode tmp = head.next;
				head.next = p;
				p = head;
				head = tmp;
				cnt--;
			}

			return p;
		}

		return head;
	}

	public int divide(int dividend, int divisor) {
		if (divisor == 0 || (dividend == Integer.MIN_VALUE && divisor == -1))
			return Integer.MAX_VALUE;

		int sign = (dividend > 0 ^ divisor > 0) ? -1 : 1;
		int res = 0;
		long dividendL = Math.abs((long) dividend);
		long divisorL = Math.abs((long) divisor);

		while (dividendL >= divisorL) {
			long tmp = divisorL;
			int mul = 1;

			while (dividendL >= (tmp << 1) && (tmp << 1 > 0)) {
				tmp <<= 1;
				mul <<= 1;
			}

			dividendL -= tmp;
			res += mul;
		}

		return sign > 0 ? res : -res;
	}

	public int[][] imageSmoother(int[][] M) {
		int m = M.length;
		int n = M[0].length;

		int[][] res = new int[m][n];

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				int sum = 0;
				int cnt = 0;

				for (int k = -1; k < 2; k++) {
					for (int p = -1; p < 2; p++) {
						if (i + k >= 0 && i + k < m && j + p >= 0 && j + p < n) {
							sum += M[i + k][j + p];
							cnt++;
						}
					}
				}

				res[i][j] = sum / cnt;
			}
		}

		return res;
	}

	public int widthOfBinaryTree(TreeNode root) {
		Queue<TreeNode> queue = new LinkedList<TreeNode>();
		queue.add(root);

		Map<TreeNode, Integer> map = new HashMap<>();
		map.put(root, 0);

		int res = 0;

		while (!queue.isEmpty()) {
			int size = queue.size();
			int start = -1;
			int end = -1;
			for (int i = 0; i < size; i++) {
				TreeNode cur = queue.poll();
				if (cur.left != null) {
					if (start == -1)
						start = map.get(cur) * 2;
					else
						end = map.get(cur) * 2;
					queue.add(cur.left);
					if (end != -1)
						res = Math.max(res, end - start);
					map.put(cur.left, map.get(cur) * 2);
				}
				if (cur.right != null) {
					if (start == -1)
						start = map.get(cur) * 2 + 1;
					else
						end = map.get(cur) * 2 + 1;
					queue.add(cur.right);
					if (end != -1)
						res = Math.max(res, end - start);
					map.put(cur.right, map.get(cur) * 2 + 1);
				}
			}
		}

		return res + 1;
	}

	public List<Integer> findSubstring(String s, String[] words) {
		Map<String, Integer> map = new HashMap<>();

		for (int i = 0; i < words.length; i++) {
			map.put(words[i], map.getOrDefault(words[i], 0) + 1);
		}

		List<Integer> res = new LinkedList<>();

		int k = words[0].length();
		int cnt = words.length;

		for (int i = 0; i < k; i++) {
			int left = i;
			int count = 0;
			Map<String, Integer> occur = new HashMap<>();

			for (int j = i; j <= s.length() - k; j += k) {
				String cur = s.substring(j, j + k);

				if (map.containsKey(cur)) {
					occur.put(cur, occur.getOrDefault(cur, 0) + 1);
					if (occur.get(cur) <= map.get(cur))
						count++;
					else {
						while (occur.get(cur) > map.get(cur)) {
							String tmp = s.substring(left, left + k);
							left += k;
							occur.put(tmp, occur.get(tmp) - 1);
							if (occur.get(tmp) < map.get(tmp))
								count--;
						}
					}

					if (count == cnt) {
						res.add(left);
						String str = s.substring(left, left + k);
						occur.put(str, occur.get(str) - 1);
						left += k;
						count--;
					}

				} else {
					count = 0;
					occur.clear();
					left = j + k;
				}
			}
		}

		return res;
	}

	public int longestValidParentheses(String s) {
		int[] dp = new int[s.length() + 1];

		int res = 0;

		for (int i = 2; i <= s.length(); i++) {
			if (s.charAt(i) == ')' && s.charAt(i - 1) == '(')
				dp[i] = dp[i - 2] + 2;
			else if (s.charAt(i) == ')' && s.charAt(i - 1) == ')') {
				if (s.charAt(i - dp[i - 1] - 1) == '(')
					dp[i] = dp[i - 1] + dp[i - dp[i - 1] - 2] + 2;
			}

			res = Math.max(res, dp[i]);
		}

		return res;
	}

	public int search(int[] nums, int target) {
		if (nums.length == 0)
			return -1;

		int low = 0, high = nums.length - 1;

		while (low < high) {
			int mid = (high - low) / 2 + low;
			if (nums[mid] > nums[high])
				low = mid + 1;
			else
				high = mid;
		}

		int min = low;
		if (target > nums[0]) {
			low = 0;
			high = (min - 1 >= 0) ? min - 1 : nums.length - 1;
		} else {
			low = min;
			high = nums.length - 1;
		}

		while (low <= high) {
			int mid = (high - low) / 2 + low;
			if (target < nums[mid])
				high = mid - 1;
			else if (target > nums[mid])
				low = mid + 1;
			else
				return mid;
		}

		if (high >= 0 && target == nums[high])
			return high;
		else
			return -1;

	}

	public void solveSudoku(char[][] board) {
		boolean[][] box = new boolean[9][9];
		boolean[][] row = new boolean[9][9];
		boolean[][] col = new boolean[9][9];

		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 9; j++) {
				int boxNum = i / 3 * 3 + j / 3;
				if (board[i][j] != '.') {
					int k = board[i][j] - '1';
					box[boxNum][k] = true;
					row[i][k] = true;
					col[k][j] = true;
				}
			}
		}

		solveSudokuHelper(board, box, row, col);
	}

	private boolean solveSudokuHelper(char[][] board, boolean[][] box, boolean[][] row, boolean[][] col) {
		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 9; j++) {
				int boxNum = i / 3 * 3 + j / 3;
				if (board[i][j] == '.') {
					for (int k = 0; k < 9; k++) {
						if (solveSudokuIsValid(i, j, k, box, row, col)) {
							box[boxNum][k] = true;
							row[i][k] = true;
							col[k][j] = true;
							board[i][j] = (char) ('1' + k);
							if (solveSudokuHelper(board, box, row, col))
								return true;
							board[i][j] = '.';
							box[boxNum][k] = false;
							row[i][k] = false;
							col[k][j] = false;

						}
					}
					return false;
				}
			}
		}
		return true;
	}

	private boolean solveSudokuIsValid(int i, int j, int k, boolean[][] box, boolean[][] row, boolean[][] col) {
		int boxNum = i / 3 * 3 + j / 3;
		if (box[boxNum][k] || row[i][k] || col[k][j])
			return false;
		return true;
	}

	public int firstMissingPositive(int[] nums) {
		for (int i = 0; i < nums.length; i++) {
			while (nums[i] > 0 && nums[i] <= nums.length && nums[i] != nums[nums[i] - 1]) {
				int tmp1 = nums[i];
				int tmp2 = nums[nums[i] - 1];
				nums[nums[i] - 1] = tmp1;
				nums[i] = tmp2;

			}
		}

		for (int i = 0; i < nums.length; i++) {
			if (nums[i] != i + 1)
				return i + 1;
		}

		return nums.length + 1;
	}

	public boolean isMatchWildCard(String s, String p) {
		/**** Two Pointer Solution (Accepted) ******/
		int i = 0, j = 0;
		int starp = -1;
		int stars = -1;

		while (i < s.length()) {
			if (j < p.length() && (s.charAt(i) == p.charAt(j) || p.charAt(j) == '?')) {
				i++;
				j++;
				continue;
			}

			if (j < p.length() && p.charAt(j) == '*') {
				starp = j;
				stars = i;
				j++;
				continue;
			}

			if (starp != -1) {
				j = starp + 1;
				stars++;
				i = stars;
				continue;
			}

			return false;
		}

		while (j < p.length() && p.charAt(j) == '*')
			j++;

		if (i == s.length() && j == p.length())
			return true;
		return false;

		/**** DP Solution (Accepted) ******/
		// boolean[][] dp = new boolean[s.length() + 1][p.length() + 1];
		// dp[0][0] = true;
		//
		// for(int i = 1; i <= p.length(); i++) {
		// dp[0][i] = (p.charAt(i - 1) == '*' && dp[0][i - 1]);
		// }
		//
		// for(int i = 1; i <= s.length(); i++) {
		// for(int j = 1; j <= p.length(); j++) {
		// if(p.charAt(j - 1) == '?' || s.charAt(i - 1) == p.charAt(j - 1))
		// dp[i][j] = dp[i - 1][j - 1];
		// else if(p.charAt(j - 1) == '*') dp[i][j] = dp[i - 1][j] || dp[i][j -
		// 1];
		// }
		// }
		// return dp[s.length()][p.length()];

		/**** Recursive Solution (Time Limit Exceeded) ******/
		// int i = 0, j = 0;
		//
		// if (s.isEmpty()) {
		// for (int k = 0; k < p.length(); k++) {
		// if (p.charAt(k) != '*')
		// return false;
		// }
		// return true;
		// }
		//
		// if(p.isEmpty()) return false;
		//
		// if (p.charAt(j) == '?') {
		// return isMatchWildCard(s.substring(i + 1), p.substring(j + 1));
		// } else if (p.charAt(j) == '*') {
		// if (j == p.length() - 1)
		// return true;
		// else {
		// while (i < s.length()) {
		// if (s.charAt(i) == p.charAt(j + 1) || p.charAt(j + 1) == '?' ||
		// p.charAt(j + 1) == '*') {
		// if (isMatchWildCard(s.substring(i), p.substring(j + 1)))
		// return true;
		// }
		// i++;
		// }
		// return false;
		// }
		// } else {
		// if (s.charAt(i) != p.charAt(j))
		// return false;
		// return isMatchWildCard(s.substring(i + 1), p.substring(j + 1));
		// }

	}

	public int jump(int[] nums) {
		if (nums.length < 2)
			return 0;

		int level = 0, i = 0;
		int curMax = 0, nextMax = 0;

		while (curMax - i + 1 > 0) {
			level++;
			for (; i <= curMax; i++) {
				nextMax = Math.max(nextMax, nums[i] + i);
				if (nextMax >= nums.length - 1)
					return level;
			}
			curMax = nextMax;
		}

		return 0;
	}

	public List<List<Integer>> permuteUnique(int[] nums) {
		Arrays.sort(nums);

		List<List<Integer>> res = new LinkedList<>();
		boolean[] used = new boolean[nums.length];

		if (nums.length == 0)
			return res;

		permuteUniqueHelper(nums, res, used, new LinkedList<Integer>());

		return res;

	}

	private void permuteUniqueHelper(int[] nums, List<List<Integer>> res, boolean[] used, List<Integer> cur) {
		if (cur.size() == nums.length)
			res.add(new LinkedList<>(cur));

		for (int i = 0; i < nums.length; i++) {
			if (!used[i]) {
				used[i] = true;
				cur.add(nums[i]);
				permuteUniqueHelper(nums, res, used, cur);
				used[i] = false;
				cur.remove(cur.size() - 1);
				while (i < nums.length - 1 && nums[i] == nums[i + 1])
					i++;
			}

		}
	}

	public double myPow(double x, int n) {
		double res = 1;
		int sign = 1;
		boolean toAdd = false;
		
		if(n == Integer.MIN_VALUE) {
			toAdd = true;
			n = Integer.MAX_VALUE;
			sign = -1;
		} else if (n < 0) {
			sign = -1;
			n = Math.abs(n);
		}

		int mul = 0;
		while (mul < n) {
			int cur = 1;
			double tmp = x;
			while (cur <= (n - mul) / 2) {
				tmp *= tmp;
				cur = cur * 2;
			}
			mul += cur;
			res *= tmp;
		}
		
		if(toAdd) res *= x;

		return sign == 1 ? res : 1 / res;
	}
	
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> res = new LinkedList<>();
        boolean[][] board = new boolean[n][n];
        
        solveNQueensHelper(res, new LinkedList<>(), board, 0);
        
        return res;
    }
    
    private void solveNQueensHelper(List<List<String>> res, List<String> cur, boolean[][] board, int row) {
    	int n = board.length;
    	
    	if(row == n) res.add(new LinkedList<>(cur));
    	
    	for(int i = 0; i < n; i++) {
    		if(solveNQueensIsValid(board, row, i)) {
    			board[row][i] = true;
    			String curStr = "";
    			int j = 0;
    			for(; j < i; j++) curStr += ".";
    			curStr += "Q";
    			for(j = i + 1; j < n; j++) curStr += ".";
    			cur.add(curStr);
    			solveNQueensHelper(res, cur, board, row + 1);
    			cur.remove(cur.size() - 1);
    			board[row][i] = false;
    		}
    	}
    }
    
    private boolean solveNQueensIsValid(boolean[][] board, int row, int col) {
    	int n = board.length;
    	
    	for(int i = 0; i < board.length; i++) {
    		if(board[i][col]) return false;
    	}
    	
    	int i = 0, j = 0;
    	while(row - i >= 0 && col - j >= 0) {
    		if(board[row - i][col - j]) return false;
    		i++; j++;
    	}
    	
    	i = 0; j = 0;
    	while(row - i >= 0 && col + j < n) {
    		if(board[row - i][col + j]) return false;
    		i++; j++;
    	}
    	
    	return true;
    }

}
