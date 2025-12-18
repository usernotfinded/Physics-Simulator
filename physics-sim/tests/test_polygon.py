import numpy as np
import pytest
from physics_sim.types import RigidBody2D, Box, ConvexPolygon
from physics_sim.scene import Scene
from physics_sim.io.json_io import body_to_json, body_from_json
from physics_sim.collision.broadphase import aabb_for_body

def test_polygon_inertia_vs_box():
    """Verify that a square polygon has same inertia as equivalent Box."""
    mass = 12.0
    hx, hy = 1.0, 1.0
    
    # Box inertia: 1/12 * m * (w^2 + h^2)
    # w=2, h=2 -> 4+4=8. -> 1/12 * 12 * 8 = 8.0
    box_body = RigidBody2D(Box((hx, hy)), mass=mass)
    expected_inertia = box_body.inertia
    
    # Equivalent polygon (CCW)
    verts = np.array([
        [-1.0, -1.0],
        [ 1.0, -1.0],
        [ 1.0,  1.0],
        [-1.0,  1.0]
    ], dtype=np.float64)
    poly_body = RigidBody2D(ConvexPolygon(verts), mass=mass)
    
    # Allow small float error
    assert abs(poly_body.inertia - expected_inertia) < 1e-9

def test_polygon_aabb():
    """Verify AABB of a rotated triangle."""
    # Triangle: (0,0), (2,0), (0,2) approx center (0.66, 0.66)
    # Let's use a centered diamond for easier math.
    # (1,0), (0,1), (-1,0), (0,-1)
    verts = np.array([
        [ 1.0,  0.0],
        [ 0.0,  1.0],
        [-1.0,  0.0],
        [ 0.0, -1.0]
    ])
    poly = ConvexPolygon(verts)
    body = RigidBody2D(poly, mass=1.0, position=(10.0, 10.0), angle=0.0)
    
    # Unrotated AABB: [9, 9, 11, 11]
    aabb = aabb_for_body(body)
    assert aabb == pytest.approx((9.0, 9.0, 11.0, 11.0), abs=1e-9)
    
    # Rotate 45 deg. (1,0) becomes (0.707, 0.707)
    # Max coordinate should be approx 0.707? No wait.
    # Diamond rotated 45 deg becomes a Square aligned with axes?
    # Vertices at distance 1 from origin.
    # A square with vertices at (+-1, 0) rotated 45 deg -> vertices at (+-0.707, +-0.707)? 
    # No, rotated 45 deg, the vertices (1,0) -> (cos45, sin45) = (0.707, 0.707).
    # So the AABB should shrink to [-0.707, 0.707] range (plus position).
    
    body.angle = np.pi / 4.0
    aabb_rot = aabb_for_body(body)
    
    val = np.cos(np.pi/4) # 0.707106...
    # Expected min/max relative to 10.0
    expected = (10.0 - val, 10.0 - val, 10.0 + val, 10.0 + val)
    
    assert aabb_rot == pytest.approx(expected, abs=1e-6)

def test_polygon_io():
    """Verify JSON round-trip for polygon."""
    verts = np.array([[0,0], [1,0], [0,1]], dtype=np.float64)
    p = ConvexPolygon(verts)
    b = RigidBody2D(p, mass=5.0, position=(1,2), angle=0.5)
    
    data = body_to_json(b)
    
    assert data["shape"]["type"] == "polygon"
    assert len(data["shape"]["vertices"]) == 3
    
    b2 = body_from_json(data)
    assert isinstance(b2.shape, ConvexPolygon)
    assert len(b2.shape.vertices) == 3
    assert b2.mass == 5.0
    
    # Check vertices match
    v2 = b2.shape.vertices
    assert np.allclose(v2, verts)

def test_polygon_collision():
    """Verify collision between two polygons (rectangles)."""
    # Two squares colliding
    # Body A: centered at 0, radius approx 1 (2x2 box equivalent)
    verts = np.array([[-1,-1], [1,-1], [1,1], [-1,1]], dtype=np.float64)
    
    a = RigidBody2D(ConvexPolygon(verts), mass=1.0, position=(0,0))
    b = RigidBody2D(ConvexPolygon(verts), mass=1.0, position=(1.5, 0)) # Overlap of 0.5 horizontally
    
    scene = Scene()
    scene.add_body(a)
    scene.add_body(b)
    
    contacts = scene._contacts()
    assert len(contacts) > 0
    c = contacts[0]
    assert c.penetration > 0.0
    
    # Separate them
    b.position = np.array([3.0, 0.0], dtype=np.float64)
    contacts = scene._contacts()
    assert len(contacts) == 0
