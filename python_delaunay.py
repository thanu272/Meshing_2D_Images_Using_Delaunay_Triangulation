
import sys, os, math

# Validation Constants
MIN_COORDINATE = -100000
MAX_COORDINATE = 100000
MIN_POINTS_FOR_TRIANGULATION = 3
MAX_POINTS_FOR_TRIANGULATION = 100000
EPSILON = 1e-10  # For floating point comparisons

def validate_coordinate(value, param_name="coordinate"):
    """Validate coordinate value is within acceptable range"""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{param_name} must be numeric, got {type(value)}")
    
    if not (MIN_COORDINATE <= value <= MAX_COORDINATE):
        raise ValueError(f"{param_name} {value} out of range [{MIN_COORDINATE}, {MAX_COORDINATE}]")
    
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{param_name} is NaN or infinite: {value}")
    
    return True

def validate_triangle_vertices(tri):
    """Validate triangle has three distinct points"""
    if not isinstance(tri, (list, tuple)) or len(tri) != 3:
        raise ValueError(f"Triangle must have exactly 3 vertices, got {len(tri)}")
    
    for i, vertex in enumerate(tri):
        if not isinstance(vertex, (list, tuple)) or len(vertex) != 2:
            raise ValueError(f"Vertex {i} must be a 2D point, got {vertex}")
        
        validate_coordinate(vertex[0], f"vertex[{i}][x]")
        validate_coordinate(vertex[1], f"vertex[{i}][y]")
    
    return True

#Function for determining the circumcircle of any three points
def circumcircle(tri):
    """Calculate circumcircle of a triangle with validation"""
    try:
        validate_triangle_vertices(tri)
        
        D = ( (tri[0][0] - tri[2][0]) * (tri[1][1] - tri[2][1]) - (tri[1][0] -  tri[2][0]) * (tri[0][1] - tri[2][1]) )
        
        # Check for degenerate triangle (collinear points)
        if abs(D) < EPSILON:
            return None
        
        center_x = (((tri[0][0] - tri[2][0]) * (tri[0][0] + tri[2][0]) + (tri[0][1] - tri[2][1]) * (tri[0][1] + tri[2][1])) / 2 * (tri[1][1] - tri[2][1]) - ((tri[1][0] - tri[2][0]) * (tri[1][0] + tri[2][0]) + (tri[1][1] - tri[2][1]) * (tri[1][1] + tri[2][1])) / 2 * (tri[0][1] - tri[2][1])) / D
        
        center_y = (((tri[1][0] - tri[2][0]) * (tri[1][0] + tri[2][0]) + (tri[1][1] - tri[2][1]) * (tri[1][1] + tri[2][1])) / 2 * (tri[0][0] - tri[2][0]) - ((tri[0][0] - tri[2][0]) * (tri[0][0] + tri[2][0]) + (tri[0][1] - tri[2][1]) * (tri[0][1] + tri[2][1])) / 2 * (tri[1][0] - tri[2][0])) / D
        
        # Validate center coordinates
        validate_coordinate(center_x, "circumcircle center_x")
        validate_coordinate(center_y, "circumcircle center_y")
        
        radius = math.sqrt ((tri[2][0] - center_x)**2 + (tri[2][1] - center_y)**2 )
        
        if radius < 0:
            raise ValueError(f"Calculated negative radius: {radius}")
        
        if math.isnan(radius) or math.isinf(radius):
            return None
        
        return [[center_x, center_y], radius]
    except (ZeroDivisionError, ValueError) as e:
        return None
    except Exception as e:
        print(f"Warning: circumcircle calculation error: {str(e)}")
        return None

#Determine if any given point lies inside a circle
def pointInCircle(point, circle):
    """Check if point is inside circle with validation"""
    try:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"Point must be 2D, got {point}")
        
        if circle is None:
            return False
        
        if not isinstance(circle, (list, tuple)) or len(circle) != 2:
            raise ValueError(f"Circle format invalid, got {circle}")
        
        validate_coordinate(point[0], "point[x]")
        validate_coordinate(point[1], "point[y]")
        validate_coordinate(circle[0][0], "circle center[x]")
        validate_coordinate(circle[0][1], "circle center[y]")
        validate_coordinate(circle[1], "circle radius")
        
        if circle[1] < 0:
            raise ValueError(f"Circle radius cannot be negative: {circle[1]}")
        
        d = math.sqrt( math.pow(point[0] - circle[0][0], 2) + math.pow(point[1] - circle[0][1], 2) )
        
        if d < circle[1]:
            return True
        else:
            return False
    except Exception as e:
        print(f"Warning: pointInCircle error: {str(e)}")
        return False
	
#Basic Point class
class Point():
	def __init__(self, x, y):
		"""Initialize point with validated coordinates"""
		try:
			validate_coordinate(x, "x")
			validate_coordinate(y, "y")
		except ValueError as e:
			raise ValueError(f"Invalid point coordinates: {str(e)}")
		
		self._x = x
		self._y = y
	
	#Position of the point
	def pos(self):
		return [self._x, self._y]
			
	#Determines if two points are equivalent
	def isEqual(self, other_point):
		if not isinstance(other_point, Point):
			return False
		if(self._x == other_point._x and self._y == other_point._y): 
			return True
		else: 
			return False
	
	#Convert the point into a string (for debugging purposes)
	def pointToStr(self):
		return str(self.pos())

#Basic Edge class
class Edge():
	def __init__(self, a, b):
		"""Initialize edge with validation"""
		if not isinstance(a, Point) or not isinstance(b, Point):
			raise TypeError("Edge points must be Point objects")
		
		if a is b:
			raise ValueError("Edge cannot connect a point to itself")
		
		if a.isEqual(b):
			raise ValueError("Edge cannot connect two identical points")
		
		self._a = a
		self._b = b
	
	#Tests if two edges are equivalent to each other
	def isEqual(self, other_edge):
		if not isinstance(other_edge, Edge):
			return False
		
		if (self._a.isEqual(other_edge._a) or self._b.isEqual(other_edge._a)) and (self._a.isEqual(other_edge._b) or self._b.isEqual(other_edge._b)):
			return True
		elif self == other_edge:
			return True
		else:
			return False
	
	#Converts an edge to a string (for debugging purposes)
	def edgeToStr(self):
		return str([self._a.pos(), self._b.pos()])
	
	#Calculate the length of an edge
	def length(self):
		try:
			length = math.sqrt( math.pow(self._b.pos()[0] - self._a.pos()[0], 2) + math.pow(self._b.pos()[1] - self._a.pos()[1], 2))
			if math.isnan(length) or math.isinf(length):
				raise ValueError(f"Invalid edge length: {length}")
			return length
		except Exception as e:
			raise RuntimeError(f"Failed to calculate edge length: {str(e)}")
	
	#Determine if two edges intersect
	def edgeIntersection(self, other_edge):
		"""Check if two edges intersect with validation"""
		try:
			if not isinstance(other_edge, Edge):
				raise TypeError("other_edge must be an Edge object")

			if self.isEqual(other_edge):
				return False
			
			try:
				x1 = self._a.pos()[0]
				x2 = self._b.pos()[0]
				x3 = other_edge._a.pos()[0]
				x4 = other_edge._b.pos()[0]
				y1 = self._a.pos()[1]
				y2 = self._b.pos()[1]
				y3 = other_edge._a.pos()[1]
				y4 = other_edge._b.pos()[1]
				
				denominator = ((x1 - x2)*(y3 - y4)) - ((y1 - y2)*(x3 - x4))
				
				# Check for parallel lines
				if abs(denominator) < EPSILON:
					return False
				
				t = (((x1 - x3)*(y3 - y4)) - ((y1 - y3)*(x3 - x4))) / denominator
				u = (((x2 - x1)*(y1 - y3)) - ((y2 - y1)*(x1 - x3))) / denominator
				
				#If 0 <= t <= 1 or 0 <= u <= 1, then an intersection occurs.
				if (t >= 0 and t <= 1) and (u >= 0 and u <= 1):
					int_x = int(x1 + t*(x2 - x1))
					int_y = int(y1 + t*(y2 - y1))
					int_point = Point(int_x, int_y)
					
					#If the intersection point is one of the edge points, then an intersection is not considered to have occurred (i.e., these are edges connected at the same point)
					if self._a.isEqual(int_point) or self._b.isEqual(int_point) or other_edge._a.isEqual(int_point) or other_edge._b.isEqual(int_point):
						return False
					
					#If there is no point, these edges intersect
					else:
						return True
					
				else:
					return False
			except ZeroDivisionError:
				#A divide-by-zero error is interpreted as the edges not intersecting
				return False
			except Exception as e:
				print(f"Warning: Edge intersection check failed: {str(e)}")
				return False
		except Exception as e:
			print(f"Warning: edgeIntersection validation error: {str(e)}")
			return False

#Basic Triangle class
class Triangle():
	
	#Cannot create a triangle if any two points are the same
	def __init__(self, a, b, c):
		"""Initialize triangle with validation"""
		if not isinstance(a, Point) or not isinstance(b, Point) or not isinstance(c, Point):
			raise TypeError("Triangle points must be Point objects")
		
		# Check all points are distinct
		if a.isEqual(b) or a.isEqual(c) or b.isEqual(c):
			raise ValueError("Triangle points must be distinct")
		
		if a is b or a is c or b is c:
			raise ValueError("Triangle cannot have duplicate point references")
		
		self._a = a
		self._b = b
		self._c = c
	
	#Test if any two triangles are equal (defined as sharing all three points)
	def isEqual(self, other_tri):
		if not isinstance(other_tri, Triangle):
			return False
		
		if (self._a is other_tri._a or self._a is other_tri._b or self._a is other_tri._c) and (self._b is other_tri._a or self._b is other_tri._b or self._b is other_tri._c) and (self._c is other_tri._a or self._c is other_tri._b or self._c is other_tri._c): 
			return True
		else: 
			return False
	
	#Prints the triangle in a neat format (for debugging purposes)
	def printTriangle(self):
		print("A: " + self._a.pointToStr() + " B: " + self._b.pointToStr() + " C: " + self._c.pointToStr())

#Graph class
class Graph():
	def __init__(self):
		
		#This will be a list of point objects as defined above
		self._points = []
		
		#This will be a list of triangle objects as defined above
		self._triangles = []
		
		#This is a list of edges as defined above
		self._edges = []
		
		#Point boundaries for sorting purposes
		self._point_min_x = 0
		self._point_max_x = 0
		
	def addPoint(self, point):
		"""Add point to graph with validation"""
		if not isinstance(point, Point):
			raise TypeError("point must be a Point object")
		
		# Limit maximum points
		if len(self._points) >= MAX_POINTS_FOR_TRIANGULATION:
			raise ValueError(f"Maximum points exceeded: {MAX_POINTS_FOR_TRIANGULATION}")
		
		#Check to see if an equivalent point exists
		for x in self._points:
			if x.isEqual(point): 
				return False
		
		try:
			#If the point has an X value lower than any other point
			if self._point_min_x > point.pos()[0] or self._point_min_x == 0:
				self._points.insert(0, point)
				self._point_min_x = point.pos()[0]
				return True
			
			#If the point has an X value higher than any other point
			elif self._point_max_x < point.pos()[0]:
				self._points.append(point)
				self._point_max_x = point.pos()[0]
				return True
			
			#If the X value is somewhere in the middle
			else:
				same_x = []
				for x in self._points:
					if x.pos()[0] == point.pos()[0]:
						same_x.append(x)
				
				#If no point has the same X value as the new point, find the first point that has a greater X value and insert the new point before it
				if len(same_x) == 0:
					first_greater = 0
					for x in self._points:
						if x.pos()[0] > point.pos()[0]:
							first_greater = self._points.index(x)
							break
					self._points.insert(first_greater, point)
					return True
				
				#If there's only one point in the graph with the same X value, compare the Y values to find which order they go in
				elif len(same_x) == 1:
					index = self._points.index(same_x[0])
					if same_x[0].pos()[1] > point.pos()[1]:
						self._points.insert(index - 1, point)
						return True
					else:
						self._points.insert(index + 1, point)
						return True
				
				#If multiple points have the same X value, find where the new point needs to go based on its Y value
				else:
					first_greater_y = 0
					for x in same_x:
						if x.pos()[1] > point.pos()[1]:
							first_greater_y = self._points.index(x)
							break
					if(first_greater_y != 0):
						self._points.insert(first_greater_y, point)
						return True
					else:
						self._points.insert(self._points.index(same_x[len(same_x) - 1]), point)
						return True
		except Exception as e:
			raise RuntimeError(f"Failed to add point to graph: {str(e)}")
		
	def addEdge(self, edge):
		"""Add edge to graph with validation"""
		if not isinstance(edge, Edge):
			raise TypeError("edge must be an Edge object")
		
		#Check for an equivalent edge in the graph, add it if one doesn't exist
		for x in self._edges:
			if x.isEqual(edge):
				return False
		
		self._edges.append(edge)
		return True
		
	#Adds a triangle to the list of triangles and returns true if successful, checking if it is equal to any other triangle. Returns false if an equivalent triangle exists
	def addTriangle(self, triangle):
		"""Add triangle to graph with validation"""
		if not isinstance(triangle, Triangle):
			raise TypeError("triangle must be a Triangle object")
		
		#First check if an equivalent triangle already exists
		for x in self._triangles:
			if x.isEqual(triangle): 
				return False
		
		#If not, we can add the triangle to the graph
		self._triangles.append(triangle)
		tri = [ triangle._a.pos(), triangle._b.pos(), triangle._c.pos() ]
		return True
		
	#Tests if a given triangle is Delaunay (i.e., no other points lie within the circumcircle of the triangle)
	def triangleIsDelaunay(self, triangle):
		"""Test if triangle satisfies Delaunay condition with validation"""
		if not isinstance(triangle, Triangle):
			raise TypeError("triangle must be a Triangle object")
		
		try:
			tri = [ triangle._a.pos(), triangle._b.pos(), triangle._c.pos() ]
			cc = circumcircle(tri)
			
			# If circumcircle is invalid (collinear points), triangle is not Delaunay
			if cc is None:
				return False
			
			for x in self._points:
				#Skip the triangle's own points
				if not (x.isEqual(triangle._a) or x.isEqual(triangle._b) or x.isEqual(triangle._c)):
					try:
						if pointInCircle(x.pos(), cc):
							return False
					except:
						return False
			
			return True
		except Exception as e:
			print(f"Warning: triangleIsDelaunay error: {str(e)}")
			return False
	
	#Generates the complete Delaunay mesh by testing every possible triangle for the Delaunay condition, then marking any edges that intersect, and removing the longer of the intersecting edges
	def generateDelaunayMesh(self):
		"""Generate Delaunay mesh with validation and error handling"""
		if len(self._points) < MIN_POINTS_FOR_TRIANGULATION:
			raise ValueError(f"Not enough points for triangulation: {len(self._points)}. Minimum: {MIN_POINTS_FOR_TRIANGULATION}")
		
		try:
			#Create every possible triangle and test it for the Delaunay condition
			triangle_count = 0
			for p1 in self._points:
				for p2 in self._points:
					for p3 in self._points:
						if not p1.isEqual(p2) and not p2.isEqual(p3) and not p3.isEqual(p1):
							try:
								test_tri = Triangle(p1, p2, p3)
								if self.triangleIsDelaunay(test_tri):
									if self.addTriangle(test_tri):
										triangle_count += 1
							except Exception as e:
								print(f"Warning: Failed to create triangle: {str(e)}")
								continue
			
			print(f"Created {triangle_count} Delaunay triangles")
			
			#One more check for the Delaunay condition (probably redundant) and then adding the edges of the triangle to the graph
			triangles_to_remove = []
			for t in self._triangles[:]:  # Create a copy for safe iteration
				if not self.triangleIsDelaunay(t):
					triangles_to_remove.append(t)
				else:
					try:
						self.addEdge(Edge(t._a, t._b))
						self.addEdge(Edge(t._b, t._c))
						self.addEdge(Edge(t._c, t._a))
					except Exception as e:
						print(f"Warning: Failed to add edge: {str(e)}")
			
			# Remove non-Delaunay triangles
			for t in triangles_to_remove:
				self._triangles.remove(t)
			
			print(f"Final triangle count: {len(self._triangles)}, Edge count: {len(self._edges)}")
			
			#Checking for intersecting edges
			bad_edges = []
			for e1 in self._edges:
				for e2 in self._edges:
					if not e1.isEqual(e2):
						try:
							if e1.edgeIntersection(e2):
								len_e1 = e1.length()
								len_e2 = e2.length()
								if len_e1 >= len_e2:
									bad_edges.append(e1)
								else:
									bad_edges.append(e2)
						except Exception as e:
							print(f"Warning: Edge intersection check failed: {str(e)}")
			
			#Removing any bad (intersecting) edges from the graph
			for x in bad_edges[:]:  # Safe iteration
				for y in self._edges[:]:
					if x.isEqual(y):
						try:
							self._edges.remove(y)
						except:
							pass
						break
			
			print(f"Removed {len(bad_edges)} intersecting edges. Final edge count: {len(self._edges)}")
			
		except Exception as e:
			raise RuntimeError(f"Delaunay mesh generation failed: {str(e)}")