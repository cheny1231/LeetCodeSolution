package com.cheny.leetCode;

public class NumMatrix {
	int[][] matrix;
	int row;
	int col;
	
    public NumMatrix(int[][] matrix) {
    	if(matrix.length > 0 && matrix[0].length > 0){
    		//initialize the sum matrix
    		row = matrix.length;
        	col = matrix[0].length;
    		
    		this.matrix = new int[row + 1][col + 1];
    		this.matrix[1][1] = matrix[0][0];
    		
    		for(int i = 2; i <= row; i++) {
    			this.matrix[i][1] = matrix[i - 1][0] + this.matrix[i - 1][1];
    		}
    		
    		for(int j = 2; j <= col; j++) {
    			this.matrix[1][j] = matrix[0][j - 1] + this.matrix[1][j - 1];
    		}
    		
    		for(int i = 2; i <= row; i++) {
    			for(int j = 2; j <= col; j++) {
    				this.matrix[i][j] = matrix[i - 1][j - 1] + this.matrix[i - 1][j] + this.matrix[i][j - 1] - this.matrix[i - 1][j - 1];
    			}
    		}
    	}
    }
    
    public int sumRegion(int row1, int col1, int row2, int col2) {
    	if(this.matrix == null) return 0;
        return this.matrix[row2 + 1][col2 + 1] - this.matrix[row2 + 1][col1] - this.matrix[row1][col2 + 1] + this.matrix[row1][col1];
    }
}
