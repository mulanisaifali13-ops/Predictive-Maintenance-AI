# --- Stage 1: Build React App ---
FROM node:18-alpine as build-stage

WORKDIR /app

# Copy package.json and install packages
COPY frontend/package.json ./
RUN npm install

# Copy source code and build
COPY frontend/ ./
RUN npm run build

# --- Stage 2: Serve via Nginx ---
FROM nginx:alpine

# Copy built assets from build-stage to Nginx default folder
COPY --from=build-stage /app/dist /usr/share/nginx/html

# Expose Nginx port
EXPOSE 80

# Run Nginx in foreground
CMD ["nginx", "-g", "daemon off;"]
