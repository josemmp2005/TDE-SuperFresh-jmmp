-- Crear usuario admin
CREATE USER admin WITH PASSWORD 'admin';

-- Crear base de datos superfresh
CREATE DATABASE superfresh OWNER admin;

-- Dar permisos
ALTER ROLE admin CREATEDB;
GRANT ALL PRIVILEGES ON DATABASE superfresh TO admin;
