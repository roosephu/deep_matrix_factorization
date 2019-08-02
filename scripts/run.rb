#!/usr/bin/ruby
# gem install colorize --user
require 'optparse'
require 'date'
require 'colorize'

options = { dry: false, template: '', name: '', n_jobs: 1, base_log_dir: '~/logs' }
grid = []
OptionParser.new do |opts|
    opts.on("--name NAME", "name") { |name| options[:name] = name }
    opts.on("--n_jobs N_JOBS", "number of jobs") { |n_jobs| options[:n_jobs] = n_jobs }
    opts.on("--dry") { options[:dry] = true }
    opts.on("--base_log_dir BASE_LOG_DIR") { |base_log_dir| options[:base_log_dir] = base_log_dir }
    opts.on("--replace PARAM", "param") { |param|
        key, values = param.split '=', 2
        values = values.split ','
        grid << values.map { |value| [key, value] }
    }
    opts.on("--template TEMPLATE", "template") { |template| options[:template] = template }
end.parse!

puts options

Dir.instance_eval do
    timestamp = DateTime.now.strftime '%Y%m%d-%H%M%S'
    raise "name can't be empty" if options[:name] == ''
    log_dir = "#{options[:base_log_dir]}/#{options[:name]}-#{timestamp}"
    puts log_dir.yellow
    grid = [['LOGDIR', log_dir]].product(*grid)

    grid.each { |args|
        cmd = options[:template].dup
        args.each do |(old, new)|
            cmd.gsub! /\b#{old}\b/, new  # an unsafe regexp constructor...
        end
        puts cmd
        if not options[:dry] then
            (1..options[:n_jobs].to_i).each do |i|
                pid = spawn cmd, :out => "/dev/null", :err => "/dev/null"
                Process.wait pid
                sleep 0.05
            end
        end
    }

end
