<h1>Ray on Azure cluster</h1>

This folder contains the files Jerome Jeannin uses to spin up a ray cluster on our Azure Directory.

<h2>If you want to run you own cluster</h2>
<ol>
    <li>copy the folder to 'your-cluster'</li>
    <li>create you own key pair</li>
    <li>change ray-config.yaml
        <ol>
            <li>change the resource-group in 'provider/resource_group'</li>
            <li>change which/how many machines in 'available_node_type'</li>
            <li>the file contains an analysis of available machines with their prices and specs (might be outdated now).</li>
            <li>link your key pair in 'file_mounts'</li>
            <li>change the setup scripts in 'head_start_ray_commands' & 'worker_start_ray_commands'</li>
        </ol>
    </li>
    <li>change your setup scripts</li>
</ol>

<h2>How to start a cluster?</h2>
<ul>
    <li>ray up setup/your-cluster/ray-config.yaml</li>
</ul>