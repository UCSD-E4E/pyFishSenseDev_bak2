# Fish Endpoint Detection

The general process is as follows:

1. Segment the fish to get a binary mask.
2. Estimate the endpoints.
3. Classify the endpoints as either head or tail.
4. Correct the endpoints.
    - Head correction.
    - Tail correction.

## 1. Fish segmentation

We currently use Fishial to segment our images.

## 2. Endpoint estimation

We use PCA to find the line of best fit.

## 3. Endpoint classification

The main idea here is that if we were to split the fish in half, the side that's more convex is the side with the tail. The current algorithm based on this assumption is as follows:

1. Convert the binary mask to a polygon.
2. Calculate the line *ab* between the estimated endpoints determined by step 2.
3. Calculate the line *ab_perp* that is perpendicular to *ab*.
4. Slice the polygon by *ab_perp* to get two halves.
5. Calculate the respective convex hulls of the two halves.
6. Find the set difference between the convex hulls and their respective halves to get two new polygons.
7. Calculate the areas of the two polygons.
8. Compare the areas. The polygon with the *greatest* area indicates the polygon half that contains the tail coordinate.
9. Assign the coordinates, *head_coord* and *tail_coord*, based on which polygon they are contained in.

![classification](https://i.ibb.co/sgR5vRM/classification.gif)

## 4. Endpoint correction

Now that we have the endpoints classified, we can apply separate heuristics to each to adjust their positions.

### Head correction

Adjusting the head coordinate is somewhat easy. If we assume that the initial endpoint estimation is generally close and the classification is correct, we just have to find the point that is the furthest end of the half. But since some fish have unexpected features towards their headpoint, it's a safe bet to isolate the region a little bit.

The idea here is that if we split the head polygon by a line parallel to *ab_perp* with its centroid being the estimated head coordinate, we can take the polygon that is closer to the outside and get the "nose" of the fish. Any point on the outer boundary of this polygon is better than—or at least the same as—the initial estimate.

The current implementation of this algorithm is as follows:

1. Calculate the convex hull of the head polygon.
2. Draw a line parallel to *ab_perp* with its centroid being *head_coord*. Call this line *headpoint_line*.
3. Draw another line parallel to *ab_perp* but this time a half-*ab*-length away from *head_coord*. Get the centroid of this line and call it *headpoint_extended*.
4. Slice the convex hull of the head polygon by *headpoint_line* and select the polygon closest to *headpoint_extended*. This is the nose polygon.
5. From *headpoint_extended*, find the closest point contained in the outer boundary of the nose polygon. This is your corrected head coordinate.

![head correction](https://i.ibb.co/Qb40fNk/headpoint-correction.gif)

### Tail correction