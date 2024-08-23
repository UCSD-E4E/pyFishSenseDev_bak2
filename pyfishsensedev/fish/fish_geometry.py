import numpy as np
import cv2
import shapely
from shapely import ops

class FishGeometry:
    def __init__(self, mask: np.ndarray):
        self.mask = mask
        self.perimeter = None
        self.estimated_endpoints = None # the first estimated endpoints (client sets this)
        self.ab = None # the line connecting the endpoints 
        self.ab_perp = None # the line perpendicular to ab 
        self.polygon = None # a polygonal representation of the fish mask 
        self.halves = None # two polygon halves sliced by ab_perp 
        self.halves_convex = None # the respective convex hulls of the halves
        self.tail_coord = None # the estimated tail coord from PCA after binary classification (client sets this)
        self.head_coord = None # the estimated head coord from PCA after binary classification (clients sets this)
        self.tail_poly = None # the polygon half containing tail_coord (client sets this)
        self.head_poly = None # the polygon half containing head_coord (client sets this)
        # ^^ The client only needs to classify either the coords or the polygons. They don't have to do both.
        # The respective counterparts are inferred automatically.
        self.headpoint_line = None # a line parallel to ab_perp with its centroid being head_coord
        self.tailpoint_line = None # a line parallel to ab_perp with its centroid being tail_coord
        self.nose_point = None # a point further out from the head_coord
        self.tail_point = None # a point further out form the tail_coord
        self.head_corrected = None # the corrected head point (client sets this)
        self.tail_corrected = None # the corrected tail point (client sets this)

    def set_estimated_endpoints(self, endpoints):
        left, right = endpoints
        self.estimated_endpoints = [np.asarray(left), np.asarray(right)]
    
    def set_tail_poly(self, tail_poly):
        self.tail_poly = tail_poly

    def set_head_poly(self, head_poly):
        self.head_poly = head_poly

    def set_tail_coord(self, tail_coord):
        self.tail_coord = np.asarray(tail_coord)

    def set_head_coord(self, head_coord):
        self.head_coord = np.asarray(head_coord)

    def set_head_corrected(self, head_corrected):
        self.head_corrected = np.asarray(head_corrected)

    def set_tail_corrected(self, tail_corrected):
        self.tail_corrected = np.asarray(tail_corrected)

    def get_estimated_endpoints(self):
        assert self.estimated_endpoints is not None, "You need to set the endpoints first! Use set_estimated_endpoints()."
        return self.estimated_endpoints
    
    def get_perimeter(self):
        if self.perimeter is not None:
            return self.perimeter
        contours, _ = cv2.findContours(self.mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.perimeter = contours[0].reshape(-1, 2)
        return self.perimeter
    
    def get_ab(self):
        if self.ab is not None:
            return self.ab
        self.ab = shapely.geometry.LineString(self.get_estimated_endpoints())
        return self.ab
    
    def get_ab_perp(self):
        if self.ab_perp is not None:
            return self.ab_perp
        ab = self.get_ab()
        perimeter = self.get_perimeter()
        vert_min = abs(perimeter[:,1].min() - ab.centroid.y)
        vert_max = abs(perimeter[:,1].max() - ab.centroid.y)
        vert_len = vert_min if vert_min > vert_max else vert_max
        ab_left = ab.parallel_offset(vert_len*1.5, 'left')
        ab_right = ab.parallel_offset(vert_len*1.5, 'right')
        self.ab_perp = shapely.geometry.LineString([ab_left.centroid, ab_right.centroid])
        return self.ab_perp
    
    def get_polygon(self):
        if self.polygon is not None:
            return self.polygon
        self.polygon = shapely.geometry.Polygon(self.get_perimeter())
        return self.polygon
    
    def get_halves(self):
        if self.halves is not None:
            return self.halves
        self.halves = ops.split(self.get_polygon(), self.get_ab_perp()).geoms
        if len(self.halves) > 2:
            self.halves = sorted(self.halves, key=lambda p: p.area, reverse=True)[:2]
        return self.halves
    
    def get_halves_convex(self):
        if self.halves_convex is not None:
            return self.halves_convex
        halves = self.get_halves()
        self.halves_convex = [halves[0].convex_hull, halves[1].convex_hull]
        return self.halves_convex
    
    def get_head_poly(self):
        if self.head_poly is not None:
            return self.head_poly
        # if the head poly is not already set, try to infer it from the coord information
        head, tail = self.__find_polys_from_matching_endpoints()
        self.set_head_poly(head)
        self.set_tail_poly(tail)
        return self.head_poly
    
    def get_tail_poly(self):
        if self.tail_poly is not None:
            return self.tail_poly
        # if the tail poly is not already set, try to infer it from the coord information
        head, tail = self.__find_polys_from_matching_endpoints()
        self.set_head_poly(head)
        self.set_tail_poly(tail)
        return self.tail_poly
    
    def get_head_coord(self, endpoints=None):
        if self.head_coord is not None:
            return self.head_coord
        # if the head coord is not already set, try to infer it from the polygon information
        head, tail = self.__find_endpoints_from_matching_polys(endpoints)
        self.set_head_coord(head)
        self.set_tail_coord(tail)
        return self.head_coord
    
    def get_tail_coord(self, endpoints=None):
        if self.tail_coord is not None:
            return self.tail_coord
        # if the tail coord is not already set, try to infer it from the polygon information
        head, tail = self.__find_endpoints_from_matching_polys(endpoints)
        self.set_head_coord(head)
        self.set_tail_coord(tail)
        return self.tail_coord
    
    def get_nose_point(self):
        if self.nose_point is not None:
            return self.nose_point
        self.__find_outer_points()
        return self.nose_point
    
    def get_tail_point(self):
        if self.tail_point is not None:
            return self.tail_point
        self.__find_outer_points()
        return self.tail_point
    
    def get_headpoint_line(self):
        if self.headpoint_line is not None:
            return self.headpoint_line
        self.__find_endpoint_lines()
        return self.headpoint_line

    def get_tailpoint_line(self):
        if self.tailpoint_line is not None:
            return self.tailpoint_line
        self.__find_endpoint_lines()
        return self.tailpoint_line
    
    def get_head_corrected(self):
        assert self.head_corrected is not None, "You need to set head_corrected. Try set_head_corrected()."
        return self.head_corrected
    
    def get_tail_corrected(self):
        assert self.tail_corrected is not None, "You need to set tail_corrected. Try set_tail_corrected()."
        return self.tail_corrected

    def __find_endpoint_lines(self):
        head_coord = self.get_head_coord()
        tail_coord = self.get_tail_coord()
        ab = self.get_ab()
        ab_perp = self.get_ab_perp()
        # get the perpendicular line of ab from the headpoint
        half_len = shapely.distance(ab, shapely.Point(head_coord).centroid)
        endpoint_line1 = ab_perp.parallel_offset(half_len, 'right')
        endpoint_line2 = ab_perp.parallel_offset(half_len, 'left')
        # again choose the right line
        if (shapely.distance(endpoint_line1.centroid, shapely.Point(head_coord)) < 
            shapely.distance(endpoint_line1.centroid, shapely.Point(tail_coord))):
            self.headpoint_line = endpoint_line1
            self.tailpoint_line = endpoint_line2
        else:
            self.headpoint_line = endpoint_line2
            self.tailpoint_line = endpoint_line1
        
    def __find_outer_points(self):
        ab_perp = self.get_ab_perp()
        perimeter = self.get_perimeter()
        # draw two lines parallel to ab_perp a little bit aways from the original endpoints
        hor_min = abs(perimeter[:,0].min() - ab_perp.centroid.x)
        hor_max = abs(perimeter[:,0].max() - ab_perp.centroid.x)
        hor_len = hor_min if hor_min > hor_max else hor_max
        nose_line1 = ab_perp.parallel_offset(hor_len*2.0, 'right') # hor_len*2.0
        nose_line2 = ab_perp.parallel_offset(hor_len*2.0, 'left') # hor_len*2.0
        # choose the line near the head and get the centroid 
        if (shapely.distance(nose_line1.centroid, shapely.Point(self.get_head_coord())) < 
            shapely.distance(nose_line1.centroid, shapely.Point(self.get_tail_coord()))):
            self.nose_point = nose_line1.centroid
            self.tail_point = nose_line2.centroid
        else:
            self.nose_point = nose_line2.centroid
            self.tail_point = nose_line1.centroid
    
    def __find_endpoints_from_matching_polys(self, endpoints=None):
        assert self.head_poly is not None and self.tail_poly is not None, "You need to classify the polygons. Try set_head_poly() or set_tail_poly()."
        left_coord, right_coord = self.estimated_endpoints if endpoints == None else endpoints
        if (abs(shapely.distance(shapely.Point(left_coord), self.get_head_poly())) <
            abs(shapely.distance(shapely.Point(left_coord), self.get_tail_poly()))):
            head_coord = left_coord
            tail_coord = right_coord
        else:
            head_coord = right_coord
            tail_coord = left_coord
        return head_coord, tail_coord
    
    def __find_polys_from_matching_endpoints(self):
        assert self.head_coord is not None and self.tail_coord is not None, "You need to classify the endpoints. Try set_head_coord() or set_tail_coord()."
        halves = set.get_halves()
        if (abs(shapely.distance(shapely.Point(self.head_coord), halves[0])) <
            abs(shapely.distance(shapely.Point(self.head_coord), halves[1]))):
            head_poly = halves[0]
            tail_poly = halves[1]
        else:
            head_poly = halves[1]
            tail_poly = halves[0]
        return head_poly, tail_poly