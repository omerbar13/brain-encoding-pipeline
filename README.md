# Smart Focus Tracker

Smart Focus Tracker is a full-stack task and focus-session tracking application built with Node.js, Express, and MongoDB.

The project demonstrates a clean backend architecture using RESTful API design, separation of concerns, persistent database storage, and scalable project structure.

## Project Overview

The application provides a backend API for managing tasks and focus-related productivity data. It was built as a practical full-stack development project to demonstrate backend design, API development, database integration, and maintainable application structure.

## Tech Stack

- Node.js
- Express.js
- MongoDB Atlas
- Mongoose
- JavaScript
- dotenv

## Features

- Create tasks
- Retrieve all tasks
- Update existing tasks
- Delete tasks
- Store data persistently using MongoDB
- Structured backend architecture with routes, controllers, services, and models

## Project Structure

```text
backend/
├── src/
│   ├── controllers/      # Handles HTTP request logic
│   ├── services/         # Business logic and database operations
│   ├── models/           # Mongoose schemas
│   ├── routes/           # API route definitions
│   └── app.js            # Application entry point
├── package.json
├── package-lock.json
└── .env                  # Local environment variables, not committed
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/smart-focus-tracker.git
cd smart-focus-tracker/backend
```

### 2. Install dependencies

```bash
npm install
```

### 3. Configure environment variables

Create a `.env` file inside the `backend/` folder:

```env
MONGO_URI=mongodb+srv://<username>:<password>@cluster.mongodb.net/taskdb
```

### 4. Start the server

```bash
node src/app.js
```

The server runs locally at:

```text
http://localhost:5000
```

## API Endpoints

### Get all tasks

```http
GET /tasks
```

Example response:

```json
[
  {
    "_id": "661f123abc...",
    "title": "Study backend architecture",
    "createdAt": "2026-04-30T10:00:00.000Z",
    "updatedAt": "2026-04-30T10:00:00.000Z"
  }
]
```

### Create a task

```http
POST /tasks
```

Request body:

```json
{
  "title": "Study backend architecture"
}
```

Example error response:

```json
{
  "error": "Title is required"
}
```

### Update a task

```http
PUT /tasks/:id
```

Request body:

```json
{
  "title": "Updated task title"
}
```

### Delete a task

```http
DELETE /tasks/:id
```

## Architecture

The backend follows a layered structure:

```text
Routes → Controllers → Services → Models
```

This structure separates API routing, request handling, business logic, and database interaction. The goal is to keep the codebase easier to maintain, test, and extend.

## Current Status

The current version implements the backend API and MongoDB persistence layer. Future development could include authentication, user accounts, task categories, a React frontend dashboard, and deployment.

## Future Improvements

- User authentication with JWT
- User accounts and protected routes
- Task categories or tags
- Focus-session analytics
- Frontend dashboard
- Deployment to a cloud platform

## Author

Omer Bar