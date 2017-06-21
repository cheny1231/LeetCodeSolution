package com.cheny.leetCode;

import java.util.Random;

public class LinkedListRandomNode {
	ListNode head;
	
	/** 
	 * Given a singly linked list, return a random node's value from the linked list. 
	 * Each node must have the same probability of being chosen.
	 * 
	 * @param head The linked list's head.
    Note that the head is guaranteed to be not null, so it contains at least one node. */
	public LinkedListRandomNode(ListNode head) {
		this.head = head;
	}
	
	 /** Returns a random node's value. */
    public int getRandom() {
        Random rnd = new Random();
        ListNode p = head;
        int r = p.val;
        for(int i = 1; p.next != null; i++){
        	p = p.next;
        	if(rnd.nextInt(i+1) == i)
        		r = p.val;     	
        }
        return r;
    }
}
