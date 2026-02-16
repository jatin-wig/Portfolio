
import React from 'react';

const skills = [
  { name: 'Machine Learning', icon: '🧠' },
  { name: 'Deep Learning', icon: '🔄' },
  { name: 'NLP', icon: '💬' },
  { name: 'Computer Vision', icon: '👁️' },
  { name: 'TensorFlow', icon: '📊' },
  { name: 'NumPy', icon: '🔢' },
  { name: 'Scikit-learn', icon: '🧪' },
  { name: 'Data Analysis', icon: '🗂️' },
];

const Superpowers = () => {
  return (
    <section className="py-16 px-6 bg-gray-50">
      <div className="max-w-6xl mx-auto">
        <h2 className="text-3xl font-bold text-center mb-2">My Superpowers</h2>
        <p className="text-gray-600 text-center mb-12">
          The tools and technologies I've mastered to build intelligent solutions.
        </p>
        
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {skills.map((skill) => (
            <div key={skill.name} className="skill-card">
              <span className="text-3xl">{skill.icon}</span>
              <span className="font-medium">{skill.name}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Superpowers;
