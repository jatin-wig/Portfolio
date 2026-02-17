
import React from 'react';
import { Link } from 'react-router-dom';

const Footer = () => {
  return (
    <footer className="bg-gray-900 text-white py-12 px-6">
      <div className="max-w-6xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <Link to="/" className="text-xl font-bold text-brand-purple">Jatin Wig</Link>
            <p className="mt-4 text-gray-400">
              AI/ML expert focused on building intelligent systems that solve real-world problems.
            </p>
          </div>

          <div>
            <h3 className="text-lg font-medium mb-4">Quick Links</h3>
            <ul className="space-y-2">
              <li><Link to="/" className="text-gray-400 hover:text-white transition-colors">Home</Link></li>
              <li><Link to="/skills" className="text-gray-400 hover:text-white transition-colors">Skills</Link></li>
              <li><Link to="/projects" className="text-gray-400 hover:text-white transition-colors">Projects</Link></li>
              <li><Link to="/publications" className="text-gray-400 hover:text-white transition-colors">Research & Publications</Link></li>
              <li><Link to="/experience" className="text-gray-400 hover:text-white transition-colors">Experience</Link></li>
              <li><Link to="/volunteering" className="text-gray-400 hover:text-white transition-colors">Volunteering</Link></li>
              <li><Link to="/education" className="text-gray-400 hover:text-white transition-colors">Education</Link></li>
              <li><Link to="/resume" className="text-gray-400 hover:text-white transition-colors">Resume</Link></li>
              <li><Link to="/contact" className="text-gray-400 hover:text-white transition-colors">Contact</Link></li>
            </ul>
          </div>

          <div>
            <h3 className="text-lg font-medium mb-4">Contact</h3>
            <ul className="space-y-2">
              <li className="text-gray-400">📧 wigjatin2@gmail.com</li>
            </ul>
          </div>
        </div>

        <div className="border-t border-gray-800 mt-10 pt-6 flex flex-col md:flex-row justify-between items-center">
          <p className="text-gray-400">© 2025 Jatin Wig. All rights reserved.</p>
          <div className="flex space-x-4 mt-4 md:mt-0">
            <a href="https://linkedin.com/in/jatin-wig" className="text-gray-400 hover:text-white transition-colors">LinkedIn</a>
            <a href="https://github.com/jatin-wig" className="text-gray-400 hover:text-white transition-colors">GitHub</a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
