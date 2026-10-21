FROM nginx:1.27-alpine
# Config is copied into the image rather than bind-mounted: bind mounts from the host were unreliable under Colima.
COPY nginx.conf /etc/nginx/nginx.conf
