from dataclasses import dataclass
from typing import Optional


@dataclass
class Node():
    data: Optional[str] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    majority_label: Optional[str] = None
    column: Optional[int] = None
    func: Optional[str] = None
    y: Optional[str] = None

    def print(self, depth):
        if self.left is None or self.right is None:
            print((depth*"---") + f"Leaf node( y {self.y})")
            return
        print((depth*"---") +
              f"node( column: {self.column}, data {self.data}, y {self.y} children:")
        self.left.print(depth+1)
        self.right.print(depth+1)
