import React from 'react';
import Header from '../components/Header';
import Footer from '../components/Footer';
import { motion } from 'framer-motion';
import { Mail, Linkedin, Github } from 'lucide-react';

const ContactPage = () => {
  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      <Header />

      <main className="flex-grow flex items-center justify-center relative overflow-hidden px-6 py-20">
        {/* Background Gradients */}
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-brand-purple/20 rounded-full blur-3xl -z-10 animate-pulse-slow" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-brand-pink/20 rounded-full blur-3xl -z-10 animate-pulse-slow delay-1000" />

        <div className="max-w-4xl w-full text-center space-y-12">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            className="space-y-6"
          >
            <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-brand-purple to-brand-pink font-heading">
              Let's Create Together
            </h1>
            <p className="text-xl text-gray-700 dark:text-gray-300 max-w-2xl mx-auto leading-relaxed font-semibold">
              I'm currently available for freelance work and open to full-time opportunities.
              If you have a project that needs some creative touch, let's chat.
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-2xl mx-auto"
          >
            <a href="mailto:wigjatin2@gmail.com" className="group relative p-8 rounded-3xl bg-white dark:bg-gray-900 shadow-xl border border-purple-100 dark:border-purple-900 hover:border-brand-purple transition-all duration-300 hover:-translate-y-1">
              <Mail className="w-10 h-10 text-brand-purple mb-4 mx-auto" />
              <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2 font-heading">Email Me</h3>
              <p className="text-brand-purple font-bold">wigjatin2@gmail.com</p>
            </a>

            <a href="https://linkedin.com/in/jatin-wig" target="_blank" rel="noopener noreferrer" className="group relative p-8 rounded-3xl bg-white dark:bg-gray-900 shadow-xl border border-blue-50 dark:border-blue-900 hover:border-blue-500 transition-all duration-300 hover:-translate-y-1">
              <Linkedin className="w-10 h-10 text-blue-500 mb-4 mx-auto" />
              <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2 font-heading">Connect</h3>
              <p className="text-blue-600 font-bold">LinkedIn Profile</p>
            </a>
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="flex justify-center gap-8 pt-8"
          >
            <SocialLink href="https://github.com/jatin-wig" icon={<Github />} label="GitHub" />
          </motion.div>
        </div>
      </main>

      <Footer />
    </div>
  );
};

const SocialLink = ({ href, icon, label }: { href: string; icon: React.ReactNode; label: string }) => (
  <a
    href={href}
    target="_blank"
    rel="noopener noreferrer"
    className="text-gray-500 hover:text-white transition-colors p-2 hover:bg-white/5 rounded-full"
    title={label}
  >
    {React.cloneElement(icon as React.ReactElement, { size: 24 })}
  </a>
);

export default ContactPage;
