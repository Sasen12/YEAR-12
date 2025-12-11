-- Seed a demo user and a sample question
INSERT OR IGNORE INTO users (id, username, password_hash) VALUES (1, 'demo', '$2b$12$KIXQeYc5Z0yq2ab9u1d0Ou3Z9x7r7kE6uH9e8qN0W9Yd8bFQv1V6e');

INSERT INTO questions (subject, question_text, difficulty, question_type) VALUES
('Software Development', 'What is the primary purpose of version control?', 'easy', 'multiple_choice');

INSERT INTO answers (question_id, answer_text, is_correct) VALUES
( (SELECT id FROM questions LIMIT 1), 'To track and manage changes to source code', 1 ),
( (SELECT id FROM questions LIMIT 1), 'To compile code into binaries', 0 ),
( (SELECT id FROM questions LIMIT 1), 'To deploy applications to production', 0 );
