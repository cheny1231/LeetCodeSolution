package com.cheny.leetCode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;

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

	private int minimumTotalHelper(int min, int res, int cur, int pos, List<List<Integer>> triangle) {
		if (cur == triangle.size()) {
			if (res < min)
				return res;
			else
				return min;
		}

		List<Integer> curList = triangle.get(cur);
		res += curList.get(pos);
		min = minimumTotalHelper(min, res, cur + 1, pos, triangle);

		if (pos < curList.size() - 1) {
			res = res - curList.get(pos) + curList.get(pos + 1);
			min = minimumTotalHelper(min, res, cur + 1, pos + 1, triangle);
		}

		return min;
	}

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
		if(nums.length == 0) return 0;
		int i = 0, j = 0;
		int sum = 0;
		int min = Integer.MAX_VALUE;

		while (j < nums.length) {
			sum += nums[j++];

			while (sum >= s) {
				if(j - i < min) min = j - i;
				sum -= nums[i++];
			}
		}
		return sum == Integer.MAX_VALUE ? 0 : min;
	}
	
    public int findUnsortedSubarray(int[] nums) {
        
    }
}
