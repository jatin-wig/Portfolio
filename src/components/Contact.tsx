
import React from 'react';
import { Linkedin, Github, Twitter, Mail } from 'lucide-react';
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { motion } from 'framer-motion';

const Contact = () => {
  const handleEmailClick = () => {
    const subject = "Hiring Inquiry";
    const body = `Hello Jatin,

I am interested in hiring you. Please write your requirements below.

[Your Requirements]
[Your Projects]
[Your Timeline]
[Your Budget]
[Other Details]

Looking forward to your reply.

Best Regards,
[Your Name]`;
    
    const mailtoUrl = `mailto:wigjatin2@gmail.com?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
    window.open(mailtoUrl, '_self');
  };

  return (
    <section className="py-16 px-6">
      <div className="max-w-6xl mx-auto">
        <h2 className="text-3xl font-bold text-center mb-2">Get In Touch</h2>
        <p className="text-gray-600 dark:text-gray-400 text-center mb-12">
          Have a project in mind or just want to chat about AI? Drop me a message!
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="space-y-6"
          >
            <Card className="overflow-hidden border-0 shadow-lg">
              <CardContent className="p-0">
                <div className="bg-gradient-to-r from-brand-purple to-brand-vivid-purple p-6 text-white">
                  <h3 className="text-2xl font-bold mb-3">Contact Information</h3>
                  <p className="opacity-90 mb-6">I'd love to hear from you! Feel free to reach out through any of these channels.</p>
                </div>
                <div className="p-6 space-y-4 bg-card text-card-foreground">
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-full bg-brand-light-purple dark:bg-brand-purple/30 flex items-center justify-center">
                      <Mail className="text-brand-purple dark:text-brand-light-purple" size={18} />
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Email</p>
                      <p className="font-medium">wigjatin2@gmail.com</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-full bg-blue-50 dark:bg-blue-900/30 flex items-center justify-center">
                      <Linkedin className="text-blue-500 dark:text-blue-400" size={18} />
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">LinkedIn</p>
                      <p className="font-medium">linkedin.com/in/jatin-wig</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-full bg-gray-100 dark:bg-gray-800 flex items-center justify-center">
                      <Github className="text-gray-700 dark:text-gray-300" size={18} />
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">GitHub</p>
                      <p className="font-medium">github.com/jatin-wig</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
            
            <div className="p-6 bg-slate-900 text-white dark:bg-slate-800 rounded-lg">
              <p className="italic">
                "May your loss always converge and your gradients never vanish."
              </p>
            </div>
          </motion.div>

          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            viewport={{ once: true }}
          >
            <h3 className="text-2xl font-bold mb-6">Connect With Me</h3>
            <div className="space-y-4">
              <a href="https://linkedin.com/in/jatin-wig" className="flex items-center gap-4 p-4 border border-gray-200 dark:border-gray-800 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors">
                <div className="w-12 h-12 bg-blue-50 dark:bg-blue-900/30 rounded-full flex items-center justify-center text-blue-500">
                  <Linkedin size={24} />
                </div>
                <div>
                  <div className="font-medium">LinkedIn</div>
                  <div className="text-muted-foreground">Let's connect professionally</div>
                </div>
              </a>

              <a href="https://github.com/jatin-wig" className="flex items-center gap-4 p-4 border border-gray-200 dark:border-gray-800 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors">
                <div className="w-12 h-12 bg-gray-100 dark:bg-gray-800 rounded-full flex items-center justify-center text-gray-700 dark:text-gray-300">
                  <Github size={24} />
                </div>
                <div>
                  <div className="font-medium">GitHub</div>
                  <div className="text-muted-foreground">Check out my code repositories</div>
                </div>
              </a>

              <button 
                onClick={handleEmailClick}
                className="flex items-center gap-4 p-4 border border-gray-200 dark:border-gray-800 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors w-full text-left"
              >
                <div className="w-12 h-12 bg-red-50 dark:bg-red-900/30 rounded-full flex items-center justify-center text-red-400">
                  <Mail size={24} />
                </div>
                <div>
                  <div className="font-medium">Email</div>
                  <div className="text-muted-foreground">wigjatin2@gmail.com</div>
                </div>
              </button>
            </div>

            <div className="mt-8">
              <Button 
                className="w-full py-6 text-base"
                onClick={handleEmailClick}
              >
                <Mail className="mr-2" />
                Send me an email
              </Button>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default Contact;
