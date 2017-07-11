package com.cheny.leetCode;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

public class RandomizedCollection {
	List<Integer> list;
	Map<Integer, Set<Integer>> m;
	Random rand;

	/** Initialize your data structure here. */
	public RandomizedCollection() {
		list = new ArrayList<>();
		m = new HashMap<>();
		rand = new Random();
	}

	/**
	 * Inserts a value to the collection. Returns true if the collection did not
	 * already contain the specified element.
	 */
	public boolean insert(int val) {
		boolean res = m.containsKey(val);
		if (res)
			m.get(val).add(list.size());
		else {
			Set<Integer> s = new HashSet<Integer>();
			s.add(list.size());
			m.put(val, s);
		}
		list.add(val);
		return !res;
	}

	/**
	 * Removes a value from the collection. Returns true if the collection
	 * contained the specified element.
	 */
	public boolean remove(int val) {
		if (!m.containsKey(val))
			return false;

		int loc = m.get(val).iterator().next();
		m.get(val).remove(loc);
		if (m.get(val).isEmpty())
			m.remove(val);

		int last = list.remove(list.size() - 1);
		if (loc < list.size()) {
			list.set(loc, last);
			m.get(last).add(loc);
			m.get(last).remove(list.size());
		}

		return true;
	}

	/** Get a random element from the collection. */
	public int getRandom() {
		int loc = rand.nextInt(list.size());
		return list.get(loc);

	}

}
