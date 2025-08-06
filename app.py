# ================================================================
# IMPORTS
# ================================================================

from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
from datetime import datetime
from multiprocessing import shared_memory
import cv2
import numpy as np
import sqlite3
import os
import json
import face_recognition
import pickle
import filetype
# --- IMPORTS ---
import subprocess
from flask import flash, redirect, url_for
# ================================================================
# APPLICATION CONFIGURATION
# ================================================================
from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

# PostgreSQL configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv("SQLALCHEMY_DATABASE_URI")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = os.getenv("SQLALCHEMY_TRACK_MODIFICATIONS")
app.config['UPLOAD_FOLDER'] = os.getenv("UPLOAD_FOLDER")



# Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)


# ================================================================
# DATABASE MODELS
# ================================================================

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    # id = db.Column(db.Integer, primary_key=True)
    # username = db.Column(db.String(80), unique=True, nullable=False)
    # email = db.Column(db.String(120), unique=True)
    # password_hash = db.Column(db.String(255), nullable=False)
    # role = db.Column(db.String(20), nullable=False, default='moderator')
    # created_at = db.Column(db.DateTime, default=datetime.utcnow)
    # is_active = db.Column(db.Boolean, default=True)

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True, info={'unique_constraint_name': 'uq_users_email'})
    password_hash = db.Column(db.String(255), nullable=False)
    first_name = db.Column(db.String(50), nullable=True)
    last_name = db.Column(db.String(50), nullable=True)
    contact = db.Column(db.String(20), nullable=True)
    role = db.Column(db.String(20), nullable=False, default='moderator')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)


    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Check if provided password matches hash"""
        return check_password_hash(self.password_hash, password)

    def is_admin(self):
        """Check if user is admin"""
        return self.role == 'admin'

    def is_moderator(self):
        """Check if user is moderator or admin"""
        return self.role in ['admin', 'moderator']

    def __repr__(self):
        return f'<User {self.username}>'


from datetime import datetime
from flask import jsonify, request

# Updated Alert Model
class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    camera = db.Column(db.String(50))
    location = db.Column(db.String(100))
    time = db.Column(db.String(100))
    message = db.Column(db.String(200))
    severity = db.Column(db.String(20))
    status = db.Column(db.String(20), default='New')  # New, Acknowledged, Resolved
    is_true_detection = db.Column(db.Boolean, default=None)  # True, False, or None (unreviewed)
    reviewed_by = db.Column(db.String(50))
    reviewed_at = db.Column(db.DateTime)


class CameraSetting(db.Model):
    __tablename__ = 'camera_settings'
    
    id = db.Column(db.Integer, primary_key=True)
    source = db.Column(db.String(50), unique=True, nullable=False)
    detections = db.Column(db.JSON, nullable=False)
    object_threshold = db.Column(db.Float, nullable=False, default=0.5)
    motion_threshold = db.Column(db.Integer, nullable=False, default=30)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        """Convert model to dictionary for JSON serialization"""
        return {
            'source': self.source,
            'detections': self.detections,
            'objectThreshold': self.object_threshold,
            'motionThreshold': self.motion_threshold
        }

    def __repr__(self):
        return f'<CameraSetting {self.source}>'

# ================================================================
# LOGIN MANAGER SETUP
# ================================================================

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ================================================================
# DECORATORS
# ================================================================

def admin_required(f):
    """Decorator to require admin role"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin():
            flash('Admin access required')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

def moderator_required(f):
    """Decorator to require moderator role (admin or moderator)"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_moderator():
            flash('Moderator access required')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# ================================================================
# UTILITY FUNCTIONS
# ================================================================

FRAME_SHAPE = (240, 320, 3)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def load_camera_settings():
    """Loads camera settings from database."""
    try:
        settings = CameraSetting.query.all()
        return [setting.to_dict() for setting in settings]
    except Exception as e:
        print(f"Error loading camera settings: {e}")
        return []

def migrate_json_to_db():
    """One-time migration function to move JSON data to database."""
    json_file = "camera_settings.json"
    if os.path.exists(json_file):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                
            CameraSetting.query.delete()
            
            for camera_data in data:
                if isinstance(camera_data, dict):
                    setting = CameraSetting(
                        source=camera_data.get('source', ''),
                        detections=camera_data.get('detections', ['motion', 'object', 'face']),
                        object_threshold=camera_data.get('objectThreshold', 0.5),
                        motion_threshold=camera_data.get('motionThreshold', 30)
                    )
                    db.session.add(setting)
            
            db.session.commit()
            print(f"Successfully migrated {len(data)} camera settings to database")
            os.rename(json_file, f"{json_file}.migrated")
            
        except Exception as e:
            db.session.rollback()
            print(f"Error migrating camera settings: {e}")

def create_default_admin():
    """Create default admin user if it doesn't exist"""
    with app.app_context():
        db.create_all()
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin = User(username='admin', role='admin')
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("Default admin user created with username: admin, password: admin123")

def gen_frames(shm_name):
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
    except FileNotFoundError:
        print("Shared memory block not found. Is the backend running?")
        return
    frame_buffer = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=shm.buf)
    while True:
        frame = frame_buffer.copy()
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    shm.close()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def apply_augmentations(image):
    """Generate augmented versions of the input image."""
    augmented_images = []
    
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    augmented_images.append(blurred)
    
    low_light = cv2.convertScaleAbs(image, alpha=0.5, beta=0)
    augmented_images.append(low_light)
    
    high_light = cv2.convertScaleAbs(image, alpha=1.5, beta=30)
    augmented_images.append(high_light)
    
    return augmented_images

def update_encodings(dataset_dir="dataset", encodings_file="encodings.pickle"):
    """
    Processes all images in the dataset folder, applies augmentation,
    computes face encodings, and saves them in encodings_file.
    """
    image_paths = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if allowed_file(file):
                image_paths.append(os.path.join(root, file))
    
    known_encodings = []
    known_names = []
    
    for image_path in image_paths:
        name = os.path.basename(os.path.dirname(image_path))
        image = cv2.imread(image_path)
        if image is None:
            continue

        all_images = [image] + apply_augmentations(image)

        for img in all_images:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, boxes)
            for encoding in encodings:
                known_encodings.append(encoding)
                known_names.append(name)

    data = {"encodings": known_encodings, "names": known_names}
    with open(encodings_file, "wb") as f:
        pickle.dump(data, f)
    
    return len(known_encodings)

# ================================================================
# AUTHENTICATION ROUTES
# ================================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password) and user.is_active:
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid username or password")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))



# ================================================================
# MAIN DASHBOARD ROUTES
# ================================================================


@app.route('/live_monitoring')
@login_required
def live_monitoring():
    camera_settings = load_camera_settings()
    return render_template('live_monitoring.html', camera_ids=camera_settings)

@app.route('/alerts')
@login_required
def alerts():
    return render_template('alerts.html')

@app.route('/analytics')
def analytics():
    try:
        alerts = Alert.query.all()
        
        total_alerts = len(alerts)
        critical_alerts = len([a for a in alerts if a.severity == 'Critical'])
        active_cameras = len(set([a.camera for a in alerts if a.camera]))
        
        recent_alerts = Alert.query.order_by(Alert.id.desc()).limit(10).all()
        
        severity_counts = {}
        for alert in alerts:
            if alert.severity:
                severity = alert.severity
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        if not severity_counts:
            severity_counts = {'No Data': 1}
        
        severity_labels = list(severity_counts.keys())
        severity_data = list(severity_counts.values())
        
        camera_counts = {}
        for alert in alerts:
            if alert.camera:
                camera = alert.camera
                camera_counts[camera] = camera_counts.get(camera, 0) + 1
        
        if not camera_counts:
            camera_counts = {'No Data': 0}
        
        camera_labels = list(camera_counts.keys())
        camera_data = list(camera_counts.values())
        
        location_counts = {}
        for alert in alerts:
            if alert.location:
                location = alert.location
                location_counts[location] = location_counts.get(location, 0) + 1
        
        if not location_counts:
            location_counts = {'No Data': 0}
        
        location_labels = list(location_counts.keys())
        location_data = list(location_counts.values())
        
        from datetime import datetime, timedelta
        import json
        
        timeline_data = []
        timeline_labels = []
        
        for i in range(7):
            day = datetime.now() - timedelta(days=i)
            day_str = day.strftime('%Y-%m-%d')
            timeline_labels.append(day.strftime('%m/%d'))
            
            day_alerts = len([a for a in alerts if a.time and day_str in a.time])
            timeline_data.append(day_alerts)
        
        timeline_labels.reverse()
        timeline_data.reverse()
        
        return render_template('analytics.html',
                             total_alerts=total_alerts,
                             critical_alerts=critical_alerts,
                             active_cameras=active_cameras,
                             recent_alerts=recent_alerts,
                             severity_labels=json.dumps(severity_labels),
                             severity_data=json.dumps(severity_data),
                             camera_labels=json.dumps(camera_labels),
                             camera_data=json.dumps(camera_data),
                             location_labels=json.dumps(location_labels),
                             location_data=json.dumps(location_data),
                             timeline_labels=json.dumps(timeline_labels),
                             timeline_data=json.dumps(timeline_data))
    
    except Exception as e:
        print(f"Error in analytics route: {e}")
        return render_template('analytics.html',
                             total_alerts=0,
                             critical_alerts=0,
                             active_cameras=0,
                             recent_alerts=[],
                             severity_labels='["No Data"]',
                             severity_data='[1]',
                             camera_labels='["No Data"]',
                             camera_data='[0]',
                             location_labels='["No Data"]',
                             location_data='[0]',
                             timeline_labels='["No Data"]',
                             timeline_data='[0]')

@app.route('/settings')
@login_required
def settings():
    cameras = load_camera_settings()
    cameras_json = json.dumps(cameras)
    return render_template('settings.html', cameras_json=cameras_json)

# ================================================================
# VIDEO STREAMING ROUTES
# ================================================================

@app.route('/video_feed/<int:cam_id>')
@login_required
def video_feed(cam_id):
    shm_name = f"video_frame_shm_{cam_id}"
    return Response(gen_frames(shm_name), mimetype='multipart/x-mixed-replace; boundary=frame')

# ================================================================
# API ROUTES - DASHBOARD DATA
# ================================================================

@app.route('/toggle_system', methods=['POST'])
@admin_required
def toggle_system():
    try:
        subprocess.Popen(["python", "main.py"])
        flash("System script started successfully.")
    except Exception as e:
        flash(f"Failed to start system: {str(e)}", "danger")
    return redirect(url_for('dashboard'))
@app.route('/')
@login_required
def dashboard():
    camera_ids = [cam.source for cam in CameraSetting.query.all()]
    alert_count = Alert.query.count()
    
    # Alert breakdown
    new_alerts = Alert.query.filter_by(status='New').count()
    acknowledged_alerts = Alert.query.filter_by(status='Acknowledged').count()
    resolved_alerts = Alert.query.filter_by(status='Resolved').count()

    # User stats
    total_users = User.query.count()
    admin_count = User.query.filter_by(role='admin').count()
    moderator_count = User.query.filter(User.role.in_(['moderator', 'admin'])).count()

    # Face encodings count
    encodings_count = 0
    encodings_path = os.path.join(app.root_path, 'encodings.pickle')
    if os.path.exists(encodings_path):
        with open(encodings_path, 'rb') as f:
            data = pickle.load(f)
            encodings_count = len(data.get("encodings", []))

    return render_template("dashboard.html",
                           camera_ids=camera_ids,
                           alert_count=alert_count,
                           encodings_count=encodings_count,
                           new_alerts=new_alerts,
                           acknowledged_alerts=acknowledged_alerts,
                           resolved_alerts=resolved_alerts,
                           total_users=total_users,
                           admin_count=admin_count,
                           moderator_count=moderator_count)


# Updated API Route with filtering
@app.route('/api/alerts')
@login_required
def api_alerts():
    # Get filter parameters
    status_filter = request.args.get('status', 'all')
    severity_filter = request.args.get('severity', 'all')
    detection_filter = request.args.get('detection', 'all')
    
    # Build query
    query = Alert.query
    
    if status_filter != 'all':
        query = query.filter(Alert.status == status_filter)
    
    if severity_filter != 'all':
        query = query.filter(Alert.severity == severity_filter)
    
    if detection_filter != 'all':
        if detection_filter == 'true':
            query = query.filter(Alert.is_true_detection == True)
        elif detection_filter == 'false':
            query = query.filter(Alert.is_true_detection == False)
        elif detection_filter == 'unreviewed':
            query = query.filter(Alert.is_true_detection == None)
    
    rows = query.order_by(Alert.id.desc()).all()
    alerts_list = []
    
    for row in rows:
        alerts_list.append({
            "id": row.id,
            "camera": row.camera,
            "location": row.location,
            "time": row.time,
            "message": row.message,
            "severity": row.severity,
            "status": row.status,
            "is_true_detection": row.is_true_detection,
            "reviewed_by": row.reviewed_by,
            "reviewed_at": row.reviewed_at.isoformat() if row.reviewed_at else None
        })
    
    return jsonify(alerts_list)

# New API route to update alert status
@app.route('/api/alerts/<int:alert_id>/update', methods=['POST'])
@login_required
def update_alert(alert_id):
    alert = Alert.query.get_or_404(alert_id)
    data = request.get_json()
    
    if 'status' in data:
        alert.status = data['status']
    
    if 'is_true_detection' in data:
        alert.is_true_detection = data['is_true_detection']
        alert.reviewed_by = current_user.username
        alert.reviewed_at = datetime.now()
    
    db.session.commit()
    
    return jsonify({
        "success": True,
        "message": "Alert updated successfully",
        "alert": {
            "id": alert.id,
            "status": alert.status,
            "is_true_detection": alert.is_true_detection,
            "reviewed_by": alert.reviewed_by,
            "reviewed_at": alert.reviewed_at.isoformat() if alert.reviewed_at else None
        }
    })

# Updated store_alert function
def store_alert(camera, location, message, severity):
    alert_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_alert = Alert(
        camera=camera,
        location=location,
        time=alert_time,
        message=message,
        severity=severity,
        status='New',  # Default status
        is_true_detection=None  # Will be reviewed later
    )
    db.session.add(new_alert)
    db.session.commit()
    print(f"[INFO] Alert stored: {camera}, {location}, {alert_time}, {message}, {severity}")

# API route to get alert statistics
@app.route('/api/alerts/stats')
@login_required
def api_alert_stats():
    total_alerts = Alert.query.count()
    new_alerts = Alert.query.filter(Alert.status == 'New').count()
    acknowledged_alerts = Alert.query.filter(Alert.status == 'Acknowledged').count()
    resolved_alerts = Alert.query.filter(Alert.status == 'Resolved').count()
    
    true_detections = Alert.query.filter(Alert.is_true_detection == True).count()
    false_detections = Alert.query.filter(Alert.is_true_detection == False).count()
    unreviewed_detections = Alert.query.filter(Alert.is_true_detection == None).count()
    
    return jsonify({
        "total": total_alerts,
        "by_status": {
            "new": new_alerts,
            "acknowledged": acknowledged_alerts,
            "resolved": resolved_alerts
        },
        "by_detection": {
            "true": true_detections,
            "false": false_detections,
            "unreviewed": unreviewed_detections
        }
    })


# ================================================================
# API ROUTES - CAMERA SETTINGS
# ================================================================

@app.route('/api/camera_settings', methods=['GET'])
@login_required
def get_camera_settings():
    """Get all camera settings"""
    cameras = load_camera_settings()
    return jsonify({"status": "success", "cameras": cameras})

@app.route('/api/save_camera_settings', methods=['POST'])
@login_required
def save_camera_settings():
    """Save camera settings to database"""
    try:
        data = request.get_json()
        cameras = data.get("cameras", [])
        
        CameraSetting.query.delete()
        
        for camera_data in cameras:
            if isinstance(camera_data, dict):
                setting = CameraSetting(
                    source=camera_data.get('source', ''),
                    detections=camera_data.get('detections', ['motion', 'object', 'face']),
                    object_threshold=camera_data.get('objectThreshold', 0.5),
                    motion_threshold=camera_data.get('motionThreshold', 30)
                )
                db.session.add(setting)
        
        db.session.commit()
        
        updated_cameras = load_camera_settings()
        return jsonify({"status": "success", "cameras": updated_cameras})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/camera_settings/<source>', methods=['GET'])
@login_required
def get_single_camera_setting(source):
    """Get settings for a specific camera"""
    try:
        setting = CameraSetting.query.filter_by(source=source).first()
        if setting:
            return jsonify({"status": "success", "camera": setting.to_dict()})
        else:
            return jsonify({"status": "error", "message": "Camera not found"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/camera_settings/<source>', methods=['PUT'])
@login_required
def update_single_camera_setting(source):
    """Update settings for a specific camera"""
    try:
        data = request.get_json()
        setting = CameraSetting.query.filter_by(source=source).first()
        
        if not setting:
            setting = CameraSetting(source=source)
            db.session.add(setting)
        
        if 'detections' in data:
            setting.detections = data['detections']
        if 'objectThreshold' in data:
            setting.object_threshold = data['objectThreshold']
        if 'motionThreshold' in data:
            setting.motion_threshold = data['motionThreshold']
        
        setting.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({"status": "success", "camera": setting.to_dict()})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/camera_settings/<source>', methods=['DELETE'])
@login_required
def delete_camera_setting(source):
    """Delete settings for a specific camera"""
    try:
        setting = CameraSetting.query.filter_by(source=source).first()
        if setting:
            db.session.delete(setting)
            db.session.commit()
            return jsonify({"status": "success", "message": "Camera setting deleted"})
        else:
            return jsonify({"status": "error", "message": "Camera not found"}), 404
    except Exception as e:
        db.session.rollback()
        return jsonify({"status": "error", "message": str(e)}), 500

# ================================================================
# USER MANAGEMENT ROUTES (ADMIN ONLY)
# ================================================================

@app.route('/user_management')
@admin_required
def user_management():
    users = User.query.all()
    return render_template('user_management.html', users=users)

@app.route('/api/add_user', methods=['POST'])
@admin_required
def add_user():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    role = data.get('role', 'moderator')
    
    if not username or not password:
        return jsonify({'status': 'error', 'message': 'Username and password required'})
    
    if role not in ['admin', 'moderator']:
        return jsonify({'status': 'error', 'message': 'Invalid role. Must be admin or moderator'})
    
    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        return jsonify({'status': 'error', 'message': 'User already exists'})
    
    new_user = User(username=username, role=role)
    new_user.set_password(password)
    
    try:
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'User added successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': 'Failed to add user'})

@app.route('/api/delete_user', methods=['POST'])
@admin_required
def delete_user():
    data = request.get_json()
    username = data.get('username')
    
    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({'status': 'error', 'message': 'User not found'})
    
    if user.id == current_user.id:
        return jsonify({'status': 'error', 'message': 'Cannot delete your own account'})
    
    try:
        db.session.delete(user)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'User deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': 'Failed to delete user'})

@app.route('/api/update_user_role', methods=['POST'])
@admin_required
def update_user_role():
    data = request.get_json()
    username = data.get('username')
    new_role = data.get('role')
    
    if new_role not in ['admin', 'moderator']:
        return jsonify({'status': 'error', 'message': 'Invalid role'})
    
    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({'status': 'error', 'message': 'User not found'})
    
    if user.id == current_user.id:
        return jsonify({'status': 'error', 'message': 'Cannot change your own role'})
    
    try:
        user.role = new_role
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'User role updated successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': 'Failed to update user role'})

@app.route('/api/toggle_user_status', methods=['POST'])
@admin_required
def toggle_user_status():
    data = request.get_json()
    username = data.get('username')
    
    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({'status': 'error', 'message': 'User not found'})
    
    if user.id == current_user.id:
        return jsonify({'status': 'error', 'message': 'Cannot deactivate your own account'})
    
    try:
        user.is_active = not user.is_active
        db.session.commit()
        status = 'activated' if user.is_active else 'deactivated'
        return jsonify({'status': 'success', 'message': f'User {status} successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': 'Failed to update user status'})

# ================================================================
# FACE REGISTRATION ROUTES
# ================================================================

@app.route('/register_face', methods=['GET', 'POST'])
@login_required
def register_face():
    message = ""
    if request.method == 'POST':
        person_name = request.form.get("person_name")
        if not person_name:
            message = "Please enter the person's name."
            return render_template("register_face.html", message=message)
        
        if 'face_image' not in request.files:
            message = "No file part in the request."
            return render_template("register_face.html", message=message)
        
        file = request.files["face_image"]
        if file.filename == "":
            message = "No file selected."
            return render_template("register_face.html", message=message)
        
        print(f"Received filename: {file.filename!r}")
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
        else:
            file_bytes = file.read()
            kind = filetype.guess(file_bytes)
            print(f"Inferred image type: {kind.extension if kind else 'None'}")
            
            if kind and kind.extension in ALLOWED_EXTENSIONS:
                filename = secure_filename(f"{person_name}.{kind.extension}")
                file.stream.seek(0)
            else:
                message = "Invalid or unsupported image format."
                return render_template("register_face.html", message=message)

        person_dir = os.path.join(app.config['UPLOAD_FOLDER'], person_name)
        os.makedirs(person_dir, exist_ok=True)
        filepath = os.path.join(person_dir, filename)
        print(f"Saving to: {filepath}")
        file.seek(0)
        file.save(filepath)
        
        count = update_encodings()
        message = f"Face image saved and encodings updated. Total encodings: {count}"
    
    return render_template("register_face.html", message=message)

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page accessible without login"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        # Get form data and strip whitespaces
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        email = request.form.get('email', '').strip()
        first_name = request.form.get('firstname', '').strip()
        last_name = request.form.get('lastname', '').strip()
        contact = request.form.get('contact', '').strip()

        # Validation
        if not username or not password or not confirm_password or not email:
            flash('Please fill all required fields.', 'danger')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))

        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'danger')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
            return redirect(url_for('register'))

        # Create and save user
        new_user = User(
            username=username,
            email=email,
            first_name=first_name,
            last_name=last_name,
            contact=contact,
            role='moderator',  # Set default role
            created_at=datetime.utcnow(),
            is_active=True
        )
        new_user.set_password(password)

        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash(f'Registration failed: {str(e)}', 'danger')

    return render_template('signup.html')

    
    

# ================================================================
# APPLICATION ENTRY POINT
# ================================================================

if __name__ == '__main__':
   # create_default_admin()
    with app.app_context():
        db.create_all()
    app.run(debug=True)

